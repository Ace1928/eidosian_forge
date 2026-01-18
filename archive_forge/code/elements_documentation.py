from __future__ import annotations
import logging # isort:skip
from html import escape
from typing import TYPE_CHECKING, Any
from ..core.json_encoder import serialize_json
from ..core.templates import (
from ..document.document import DEFAULT_TITLE
from ..settings import settings
from ..util.serialization import make_globally_unique_css_safe_id
from .util import RenderItem
from .wrappers import wrap_in_onload, wrap_in_safely, wrap_in_script_tag
 Render an script for Bokeh render items.
    Args:
        docs_json_or_id:
            can be None

        render_items (RenderItems) :
            Specific items to render from the document and where

        app_path (str, optional) :

        absolute_url (Theme, optional) :

    Returns:
        str
    