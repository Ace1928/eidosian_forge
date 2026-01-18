import codecs
from pygments.util import get_bool_opt, string_types
from pygments.styles import get_style_by_name
def _lookup_style(style):
    if isinstance(style, string_types):
        return get_style_by_name(style)
    return style