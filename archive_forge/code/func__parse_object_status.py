import collections
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import param_utils
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.engine import constraints as constr
from heat.rpc import api as rpc_api
def _parse_object_status(status):
    """Parse input status into action and status if possible.

    This function parses a given string (or list of strings) and see if it
    contains the action part. The action part is exacted if found.

    :param status: A string or a list of strings where each string contains
                   a status to be checked.
    :returns: (actions, statuses) tuple, where actions is a set of actions
              extracted from the input status and statuses is a set of pure
              object status.
    """
    if not isinstance(status, list):
        status = [status]
    status_set = set()
    action_set = set()
    for val in status:
        for s in ('COMPLETE', 'FAILED', 'IN_PROGRESS'):
            index = val.rfind(s)
            if index != -1:
                status_set.add(val[index:])
                if index > 1:
                    action_set.add(val[:index - 1])
                break
    return (action_set, status_set)