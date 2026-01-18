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
def _embed_image_or_attachment(self, src: str) -> str:
    """Embed an image or attachment, depending on the configuration.
        If neither is possible, returns the original URL.
        """
    attachment_prefix = 'attachment:'
    if src.startswith(attachment_prefix):
        name = src[len(attachment_prefix):]
        if name not in self.attachments:
            msg = f'missing attachment: {name}'
            raise InvalidNotebook(msg)
        attachment = self.attachments[name]
        preferred_mime_types = ('image/svg+xml', 'image/png', 'image/jpeg')
        for mime_type in preferred_mime_types:
            if mime_type in attachment:
                return f'data:{mime_type};base64,{attachment[mime_type]}'
        default_mime_type = next(iter(attachment.keys()))
        return f'data:{default_mime_type};base64,{attachment[default_mime_type]}'
    if self.embed_images:
        base64_url = self._src_to_base64(src)
        if base64_url is not None:
            return base64_url
    return src