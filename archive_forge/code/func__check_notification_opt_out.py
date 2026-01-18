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
def _check_notification_opt_out(event_type, outcome):
    """Check if a particular event_type has been opted-out of.

    This method checks to see if an event should be sent to the messaging
    service. Any event specified in the opt-out list will not be transmitted.

    :param event_type: This is the meter name that Ceilometer uses to poll
        events. For example: identity.user.created, or
        identity.authenticate.success, or identity.role_assignment.created
    :param outcome: The CADF outcome (taxonomy.OUTCOME_PENDING,
        taxonomy.OUTCOME_SUCCESS, taxonomy.OUTCOME_FAILURE)

    """
    if 'authenticate' in event_type:
        event_type = event_type + '.' + outcome
    if event_type in CONF.notification_opt_out:
        return True
    return False