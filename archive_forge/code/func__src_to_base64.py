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
def _src_to_base64(self, src: str) -> Optional[str]:
    """Turn the source file into a base64 url.

        :param src: source link of the file.
        :return: the base64 url or None if the file was not found.
        """
    src_path = os.path.join(self.path, src)
    if not os.path.exists(src_path):
        return None
    with open(src_path, 'rb') as fobj:
        mime_type, _ = mimetypes.guess_type(src_path)
        base64_data = base64.b64encode(fobj.read())
        base64_str = base64_data.replace(b'\n', b'').decode('ascii')
        return f'data:{mime_type};base64,{base64_str}'