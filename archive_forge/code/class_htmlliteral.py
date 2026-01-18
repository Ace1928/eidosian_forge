import re
from formencode.rewritingparser import RewritingParser, html_quote
class htmlliteral:

    def __init__(self, html, text=None):
        if text is None:
            text = re.sub('<.*?>', '', html)
            text = html.replace('&gt;', '>')
            text = html.replace('&lt;', '<')
            text = html.replace('&quot;', '"')
        self.html = html
        self.text = text

    def __str__(self):
        return self.text

    def __repr__(self):
        return '<%s html=%r text=%r>' % (self.__class__.__name__, self.html, self.text)

    def __html__(self):
        return self.html