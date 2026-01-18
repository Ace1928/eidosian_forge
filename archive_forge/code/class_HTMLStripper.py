from __future__ import print_function
import re
import hashlib
class HTMLStripper(HTMLParser.HTMLParser):
    """Strip all tags from the HTML."""

    def __init__(self, collector):
        HTMLParser.HTMLParser.__init__(self)
        self.reset()
        self.collector = collector
        self.collect = True

    def handle_data(self, data):
        """Keep track of the data."""
        data = data.strip()
        if data and self.collect:
            self.collector.append(data)

    def handle_starttag(self, tag, attrs):
        HTMLParser.HTMLParser.handle_starttag(self, tag, attrs)
        if tag.lower() in ('script', 'style'):
            self.collect = False

    def handle_endtag(self, tag):
        HTMLParser.HTMLParser.handle_endtag(self, tag)
        if tag.lower() in ('script', 'style'):
            self.collect = True