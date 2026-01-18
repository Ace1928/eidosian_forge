from collections import namedtuple
import re
import textwrap
import warnings
@classmethod
def _form_media_range(cls, type_subtype, media_type_params):
    """
        Combine `type_subtype` and `media_type_params` to form a media range.

        `type_subtype` is a ``str``, and `media_type_params` is an iterable of
        (parameter name, parameter value) tuples.
        """
    media_type_params_segment = ''
    for param_name, param_value in media_type_params:
        param_value = cls._escape_and_quote_parameter_value(param_value=param_value)
        media_type_params_segment += ';' + param_name + '=' + param_value
    return type_subtype + media_type_params_segment