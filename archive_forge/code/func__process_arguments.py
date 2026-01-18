from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote_plus, urlparse
from ..core.templates import AUTOLOAD_REQUEST_TAG, FILE
from ..resources import DEFAULT_SERVER_HTTP_URL
from ..util.serialization import make_globally_unique_css_safe_id
from ..util.strings import format_docstring
from .bundle import bundle_for_objs_and_resources
from .elements import html_page_for_render_items
from .util import RenderItem
def _process_arguments(arguments: dict[str, str] | None) -> str:
    """ Return user-supplied HTML arguments to add to a Bokeh server URL.

    Args:
        arguments (dict[str, object]) :
            Key/value pairs to add to the URL

    Returns:
        str

    """
    if arguments is None:
        return ''
    result = ''
    for key, value in arguments.items():
        if not key.startswith('bokeh-'):
            result += f'&{quote_plus(str(key))}={quote_plus(str(value))}'
    return result