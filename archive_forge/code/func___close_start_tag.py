import codecs
from xml.sax.saxutils import escape, quoteattr
def __close_start_tag(self):
    if not self.closed:
        self.closed = True
        self.stream.write('>')