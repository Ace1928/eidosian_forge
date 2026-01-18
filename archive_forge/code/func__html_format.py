import logging
import sys
import weakref
import webob
from wsme.exc import ClientSideError, UnknownFunction
from wsme.protocol import getprotocol
from wsme.rest import scan_api
import wsme.api
import wsme.types
def _html_format(self, content, content_types):
    try:
        from pygments import highlight
        from pygments.lexers import get_lexer_for_mimetype
        from pygments.formatters import HtmlFormatter
        lexer = None
        for ct in content_types:
            try:
                lexer = get_lexer_for_mimetype(ct)
                break
            except Exception:
                pass
        if lexer is None:
            raise ValueError('No lexer found')
        formatter = HtmlFormatter()
        return html_body % dict(css=formatter.get_style_defs(), content=highlight(content, lexer, formatter).encode('utf8'))
    except Exception as e:
        log.warning('Could not pygment the content because of the following error :\n%s' % e)
        return html_body % dict(css='', content='<pre>%s</pre>' % content.replace(b'>', b'&gt;').replace(b'<', b'&lt;'))