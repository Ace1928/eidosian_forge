from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
@staticmethod
def _is_correct_simple_types(expected_type, value, allow_null=True):

    def is_numeric_string(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    if value is None and allow_null:
        return True
    elif expected_type == PropType.STRING:
        return isinstance(value, string_types)
    elif expected_type == PropType.BOOLEAN:
        return isinstance(value, bool)
    elif expected_type == PropType.INTEGER:
        is_integer = isinstance(value, integer_types) and (not isinstance(value, bool))
        is_digit_string = isinstance(value, string_types) and value.isdigit()
        return is_integer or is_digit_string
    elif expected_type == PropType.NUMBER:
        is_number = isinstance(value, (integer_types, float)) and (not isinstance(value, bool))
        is_numeric_string = isinstance(value, string_types) and is_numeric_string(value)
        return is_number or is_numeric_string
    return False