import copy
import re
from urllib.parse import urlsplit, unquote_plus
from lxml import etree
from lxml.html import defs
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import xhtml_to_html, _transform_result
def _has_javascript_scheme(s):
    safe_image_urls = 0
    for image_type in _find_image_dataurls(s):
        if _is_unsafe_image_type(image_type):
            return True
        safe_image_urls += 1
    return len(_possibly_malicious_schemes(s)) > safe_image_urls