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
def _load_json_or_none(text: str) -> Any:
    if isinstance(text, (str, bytes, bytearray)):
        try:
            return json.loads(text)
        except ValueError:
            return None
    return None