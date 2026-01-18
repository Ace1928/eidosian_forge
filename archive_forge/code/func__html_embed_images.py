import base64
import mimetypes
import os
from html import escape
from typing import Any, Callable, Dict, Iterable, Match, Optional, Tuple
import bs4
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexer import Lexer
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound
from nbconvert.filters.strings import add_anchor
def _html_embed_images(self, html: str) -> str:
    parsed_html = bs4.BeautifulSoup(html, features='html.parser')
    imgs: bs4.ResultSet[bs4.Tag] = parsed_html.find_all('img')
    for img in imgs:
        src = img.attrs.get('src')
        if src is None:
            continue
        base64_url = self._src_to_base64(img.attrs['src'])
        if base64_url is not None:
            img.attrs['src'] = base64_url
    return str(parsed_html)