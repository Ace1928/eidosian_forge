from .base import Style, DEFAULT_ATTRS, ANSI_COLOR_NAMES
from .defaults import DEFAULT_STYLE_EXTENSIONS
from .utils import merge_attrs, split_token_in_parts
from six.moves import range

    Turn a dictionary that maps `Token` to `Attrs` into a style class.

    :param token_to_attrs: Dictionary that maps `Token` to `Attrs`.
    