from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
@staticmethod
def _delete_empty_field_from_report(status):
    if not status[PropName.REQUIRED]:
        del status[PropName.REQUIRED]
    if not status[PropName.INVALID_TYPE]:
        del status[PropName.INVALID_TYPE]
    return status