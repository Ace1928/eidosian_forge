from html.parser import HTMLParser
class CitationParser(HTMLParser):
    """Citation Parser

    Replaces html tags with data-cite attribute with respective latex \\cite.

    Inherites from HTMLParser, overrides:
     - handle_starttag
     - handle_endtag
    """
    opentags = None
    citelist = None
    citetag = None

    def __init__(self):
        """Initialize the parser."""
        self.citelist = []
        self.opentags = 0
        HTMLParser.__init__(self)

    def get_offset(self):
        """Get the offset position."""
        lin, offset = self.getpos()
        pos = 0
        for _ in range(lin - 1):
            pos = self.data.find('\n', pos) + 1
        return pos + offset

    def handle_starttag(self, tag, attrs):
        """Handle a start tag."""
        if self.opentags == 0 and len(attrs) > 0:
            for atr, data in attrs:
                if atr.lower() == 'data-cite':
                    self.citetag = tag
                    self.opentags = 1
                    self.citelist.append([data, self.get_offset()])
                    return
        if tag == self.citetag:
            self.opentags += 1

    def handle_endtag(self, tag):
        """Handle an end tag."""
        if tag == self.citetag:
            if self.opentags == 1:
                pos = self.get_offset()
                self.citelist[-1].append(pos + len(tag) + 3)
            self.opentags -= 1

    def feed(self, data):
        """Handle a feed."""
        self.data = data
        HTMLParser.feed(self, data)