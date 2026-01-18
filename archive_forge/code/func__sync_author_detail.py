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
def _sync_author_detail(self, key='author'):
    context = self._get_context()
    detail = context.get('%ss' % key, [FeedParserDict()])[-1]
    if detail:
        name = detail.get('name')
        email = detail.get('email')
        if name and email:
            context[key] = '%s (%s)' % (name, email)
        elif name:
            context[key] = name
        elif email:
            context[key] = email
    else:
        author, email = (context.get(key), None)
        if not author:
            return
        emailmatch = re.search('(([a-zA-Z0-9\\_\\-\\.\\+]+)@((\\[[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.)|(([a-zA-Z0-9\\-]+\\.)+))([a-zA-Z]{2,4}|[0-9]{1,3})(\\]?))(\\?subject=\\S+)?', author)
        if emailmatch:
            email = emailmatch.group(0)
            author = author.replace(email, '')
            author = author.replace('()', '')
            author = author.replace('<>', '')
            author = author.replace('&lt;&gt;', '')
            author = author.strip()
            if author and author[0] == '(':
                author = author[1:]
            if author and author[-1] == ')':
                author = author[:-1]
            author = author.strip()
        if author or email:
            context.setdefault('%s_detail' % key, detail)
        if author:
            detail['name'] = author
        if email:
            detail['email'] = email