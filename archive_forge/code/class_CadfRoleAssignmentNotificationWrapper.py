import collections
import functools
import inspect
import socket
import flask
from oslo_log import log
import oslo_messaging
from oslo_utils import reflection
import pycadf
from pycadf import cadftaxonomy as taxonomy
from pycadf import cadftype
from pycadf import credential
from pycadf import eventfactory
from pycadf import host
from pycadf import reason
from pycadf import resource
from keystone.common import context
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
class CadfRoleAssignmentNotificationWrapper(object):
    """Send CADF notifications for ``role_assignment`` methods.

    This function is only used for role assignment events. Its ``action`` and
    ``event_type`` are dictated below.

    - action: ``created.role_assignment`` or ``deleted.role_assignment``
    - event_type: ``identity.role_assignment.created`` or
        ``identity.role_assignment.deleted``

    Sends a CADF notification if the wrapped method does not raise an
    :class:`Exception` (such as :class:`keystone.exception.NotFound`).

    :param operation: one of the values from ACTIONS (created or deleted)
    """
    ROLE_ASSIGNMENT = 'role_assignment'

    def __init__(self, operation):
        self.action = '%s.%s' % (operation, self.ROLE_ASSIGNMENT)
        self.event_type = '%s.%s.%s' % (SERVICE, self.ROLE_ASSIGNMENT, operation)

    def __call__(self, f):

        @functools.wraps(f)
        def wrapper(wrapped_self, role_id, *args, **kwargs):
            """Send a notification if the wrapped callable is successful.

            NOTE(stevemar): The reason we go through checking kwargs
            and args for possible target and actor values is because the
            create_grant() (and delete_grant()) method are called
            differently in various tests.
            Using named arguments, i.e.::

                create_grant(user_id=user['id'], domain_id=domain['id'],
                             role_id=role['id'])

            Or, using positional arguments, i.e.::

                create_grant(role_id['id'], user['id'], None,
                             domain_id=domain['id'], None)

            Or, both, i.e.::

                create_grant(role_id['id'], user_id=user['id'],
                             domain_id=domain['id'])

            Checking the values for kwargs is easy enough, since it comes
            in as a dictionary

            The actual method signature is

            ::

                create_grant(role_id, user_id=None, group_id=None,
                             domain_id=None, project_id=None,
                             inherited_to_projects=False)

            So, if the values of actor or target are still None after
            checking kwargs, we can check the positional arguments,
            based on the method signature.
            """
            call_args = inspect.getcallargs(f, wrapped_self, role_id, *args, **kwargs)
            inherited = call_args['inherited_to_projects']
            initiator = call_args.get('initiator', None)
            target = resource.Resource(typeURI=taxonomy.ACCOUNT_USER)
            audit_kwargs = {}
            if call_args['project_id']:
                audit_kwargs['project'] = call_args['project_id']
            elif call_args['domain_id']:
                audit_kwargs['domain'] = call_args['domain_id']
            if call_args['user_id']:
                audit_kwargs['user'] = call_args['user_id']
            elif call_args['group_id']:
                audit_kwargs['group'] = call_args['group_id']
            audit_kwargs['inherited_to_projects'] = inherited
            audit_kwargs['role'] = role_id
            try:
                result = f(wrapped_self, role_id, *args, **kwargs)
            except Exception:
                _send_audit_notification(self.action, initiator, taxonomy.OUTCOME_FAILURE, target, self.event_type, **audit_kwargs)
                raise
            else:
                _send_audit_notification(self.action, initiator, taxonomy.OUTCOME_SUCCESS, target, self.event_type, **audit_kwargs)
                return result
        return wrapper