import re
import traitlets
import datetime as dt
class NumberFormat(traitlets.Unicode):
    """A string holding a number format specifier, e.g. '.3f'

    This traitlet holds a string that can be passed to the
    `d3-format <https://github.com/d3/d3-format>`_ JavaScript library.
    The format allowed is similar to the Python format specifier (PEP 3101).
    """
    info_text = 'a valid number format'
    default_value = traitlets.Undefined

    def validate(self, obj, value):
        value = super().validate(obj, value)
        re_match = _number_format_re.match(value)
        if re_match is None:
            self.error(obj, value)
        else:
            format_type = re_match.group(9)
            if format_type is None:
                return value
            elif format_type in _number_format_types:
                return value
            else:
                raise traitlets.TraitError("The type specifier of a NumberFormat trait must be one of {}, but a value of '{}' was specified.".format(list(_number_format_types), format_type))