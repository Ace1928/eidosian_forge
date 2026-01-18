import base64
import binascii
import copy
import html.entities
import re
import xml.sax.saxutils
from .html import _cp1252
from .namespaces import _base, cc, dc, georss, itunes, mediarss, psc
from .sanitizer import _sanitize_html, _HTMLSanitizer
from .util import FeedParserDict
from .urls import _urljoin, make_safe_absolute_uri, resolve_relative_uris
def _save_contributor(self, key, value):
    context = self._get_context()
    context.setdefault('contributors', [FeedParserDict()])
    context['contributors'][-1][key] = value