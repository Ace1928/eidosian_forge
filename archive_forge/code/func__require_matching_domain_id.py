from oslo_log import log
from keystone import assignment
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.resource.backends import base
from keystone.token import provider as token_provider
def _require_matching_domain_id(self, new_ref, orig_ref):
    """Ensure the current domain ID matches the reference one, if any.

        Provided we want domain IDs to be immutable, check whether any
        domain_id specified in the ref dictionary matches the existing
        domain_id for this entity.

        :param new_ref: the dictionary of new values proposed for this entity
        :param orig_ref: the dictionary of original values proposed for this
                         entity
        :raises: :class:`keystone.exception.ValidationError`
        """
    if 'domain_id' in new_ref:
        if new_ref['domain_id'] != orig_ref['domain_id']:
            raise exception.ValidationError(_('Cannot change Domain ID'))