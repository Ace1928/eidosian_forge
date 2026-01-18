import re
from .html import _BaseHTMLProcessor
from .urls import make_safe_absolute_uri
def _sanitize_html(html_source, encoding, _type):
    p = _HTMLSanitizer(encoding, _type)
    html_source = html_source.replace('<![CDATA[', '&lt;![CDATA[')
    p.feed(html_source)
    data = p.output()
    data = data.strip().replace('\r\n', '\n')
    return data