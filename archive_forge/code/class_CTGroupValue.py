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
class CTGroupValue(TypedDict):
    _parser: Union[Type[etree.XMLParser], Type[html.HTMLParser]]
    _csstranslator: Union[GenericTranslator, HTMLTranslator]
    _tostring_method: str