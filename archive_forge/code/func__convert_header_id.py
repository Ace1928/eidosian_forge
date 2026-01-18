import base64
import os
import re
import textwrap
import warnings
from urllib.parse import quote
from xml.etree.ElementTree import Element
import bleach
from defusedxml import ElementTree  # type:ignore[import-untyped]
from nbconvert.preprocessors.sanitize import _get_default_css_sanitizer
def _convert_header_id(header_contents):
    """Convert header contents to valid id value. Takes string as input, returns string.

    Note: this may be subject to change in the case of changes to how we wish to generate ids.

    For use on markdown headings.
    """
    return quote(header_contents.replace(' ', '-'), safe="?/:@!$&'()*+,;=")