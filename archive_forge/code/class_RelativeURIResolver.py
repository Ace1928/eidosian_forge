import re
import urllib.parse
from .html import _BaseHTMLProcessor
class RelativeURIResolver(_BaseHTMLProcessor):
    relative_uris = {('a', 'href'), ('applet', 'codebase'), ('area', 'href'), ('audio', 'src'), ('blockquote', 'cite'), ('body', 'background'), ('del', 'cite'), ('form', 'action'), ('frame', 'longdesc'), ('frame', 'src'), ('iframe', 'longdesc'), ('iframe', 'src'), ('head', 'profile'), ('img', 'longdesc'), ('img', 'src'), ('img', 'usemap'), ('input', 'src'), ('input', 'usemap'), ('ins', 'cite'), ('link', 'href'), ('object', 'classid'), ('object', 'codebase'), ('object', 'data'), ('object', 'usemap'), ('q', 'cite'), ('script', 'src'), ('source', 'src'), ('video', 'poster'), ('video', 'src')}

    def __init__(self, baseuri, encoding, _type):
        _BaseHTMLProcessor.__init__(self, encoding, _type)
        self.baseuri = baseuri

    def resolve_uri(self, uri):
        return make_safe_absolute_uri(self.baseuri, uri.strip())

    def unknown_starttag(self, tag, attrs):
        attrs = self.normalize_attrs(attrs)
        attrs = [(key, (tag, key) in self.relative_uris and self.resolve_uri(value) or value) for key, value in attrs]
        super(RelativeURIResolver, self).unknown_starttag(tag, attrs)