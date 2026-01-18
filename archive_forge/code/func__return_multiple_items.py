from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
@staticmethod
def _return_multiple_items(op_params):
    """
        Defines if the operation returns one item or a list of items.

        :param op_params: operation specification
        :return: True if the operation returns a list of items, otherwise False
        """
    try:
        schema = op_params[PropName.RESPONSES][SUCCESS_RESPONSE_CODE][PropName.SCHEMA]
        return PropName.ITEMS in schema[PropName.PROPERTIES]
    except KeyError:
        return False