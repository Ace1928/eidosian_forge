import json
import typing
import warnings
from io import BytesIO
from typing import (
from warnings import warn
import jmespath
from lxml import etree, html
from packaging.version import Version
from .csstranslator import GenericTranslator, HTMLTranslator
from .utils import extract_regex, flatten, iflatten, shorten
def _get_root_from_text(text: str, *, type: str, **lxml_kwargs: Any) -> etree._Element:
    return create_root_node(text, _ctgroup[type]['_parser'], **lxml_kwargs)