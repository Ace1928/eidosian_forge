from collections import namedtuple
import re
import textwrap
import warnings
@classmethod
def _parse_media_type_params(cls, media_type_params_segment):
    """
        Parse media type parameters segment into list of (name, value) tuples.
        """
    media_type_params = cls.parameters_compiled_re.findall(media_type_params_segment)
    for index, (name, value) in enumerate(media_type_params):
        if value.startswith('"') and value.endswith('"'):
            value = cls._process_quoted_string_token(token=value)
            media_type_params[index] = (name, value)
    return media_type_params