from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
@staticmethod
def _is_object(model):
    return PropName.TYPE in model and model[PropName.TYPE] == PropType.OBJECT