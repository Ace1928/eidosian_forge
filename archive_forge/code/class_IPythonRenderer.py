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
class IPythonRenderer(HTMLRenderer):
    """An ipython html renderer."""

    def __init__(self, escape: bool=True, allow_harmful_protocols: bool=True, embed_images: bool=False, exclude_anchor_links: bool=False, anchor_link_text: str='¶', path: str='', attachments: Optional[Dict[str, Dict[str, str]]]=None):
        """Initialize the renderer."""
        super().__init__(escape, allow_harmful_protocols)
        self.embed_images = embed_images
        self.exclude_anchor_links = exclude_anchor_links
        self.anchor_link_text = anchor_link_text
        self.path = path
        if attachments is not None:
            self.attachments = attachments
        else:
            self.attachments = {}

    def block_code(self, code: str, info: Optional[str]=None) -> str:
        """Handle block code."""
        lang: Optional[str] = ''
        lexer: Optional[Lexer] = None
        if info:
            if info.startswith('mermaid'):
                return self.block_mermaidjs(code)
            try:
                if info.strip().split(None, 1):
                    lang = info.strip().split(maxsplit=1)[0]
                    lexer = get_lexer_by_name(lang, stripall=True)
            except ClassNotFound:
                code = f'{lang}\n{code}'
                lang = None
        if not lang:
            return super().block_code(code, info=info)
        formatter = HtmlFormatter()
        return highlight(code, lexer, formatter)

    def block_mermaidjs(self, code: str) -> str:
        """Handle mermaid syntax."""
        return f'<div class="jp-Mermaid"><pre class="mermaid">\n{code.strip()}\n</pre></div>'

    def block_html(self, html: str) -> str:
        """Handle block html."""
        if self.embed_images:
            html = self._html_embed_images(html)
        return super().block_html(html)

    def inline_html(self, html: str) -> str:
        """Handle inline html."""
        if self.embed_images:
            html = self._html_embed_images(html)
        return super().inline_html(html)

    def heading(self, text: str, level: int, **attrs: Dict[str, Any]) -> str:
        """Handle a heading."""
        html = super().heading(text, level, **attrs)
        if self.exclude_anchor_links:
            return html
        return str(add_anchor(html, anchor_link_text=self.anchor_link_text))

    def escape_html(self, text: str) -> str:
        """Escape html content."""
        return escape(text, quote=False)

    def block_math(self, body: str) -> str:
        """Handle block math."""
        return f'$${self.escape_html(body)}$$'

    def multiline_math(self, text: str) -> str:
        """Handle mulitline math for older mistune versions."""
        return text

    def latex_environment(self, name: str, body: str) -> str:
        """Handle a latex environment."""
        name, body = (self.escape_html(name), self.escape_html(body))
        return f'\\begin{{{name}}}{body}\\end{{{name}}}'

    def inline_math(self, body: str) -> str:
        """Handle inline math."""
        return f'${self.escape_html(body)}$'

    def image(self, text: str, url: str, title: Optional[str]=None) -> str:
        """Rendering a image with title and text.

        :param text: alt text of the image.
        :param url: source link of the image.
        :param title: title text of the image.

        :note: The parameters `text` and `url` are swapped in older versions
            of mistune.
        """
        if MISTUNE_V3:
            url = self._embed_image_or_attachment(url)
        else:
            text = self._embed_image_or_attachment(text)
        return super().image(text, url, title)

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