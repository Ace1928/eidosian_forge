from html.parser import HTMLParser
from itertools import zip_longest
class StemNode(Node):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.children = []

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def write(self, doc):
        self.collapse_whitespace()
        self._write_children(doc)

    def _write_children(self, doc):
        for child, next_child in zip_longest(self.children, self.children[1:]):
            if isinstance(child, TagNode) and next_child is not None:
                child.write(doc, next_child)
            else:
                child.write(doc)

    def is_whitespace(self):
        return all((child.is_whitespace() for child in self.children))

    def startswith_whitespace(self):
        return self.children and self.children[0].startswith_whitespace()

    def endswith_whitespace(self):
        return self.children and self.children[-1].endswith_whitespace()

    def lstrip(self):
        while self.children and self.children[0].is_whitespace():
            self.children = self.children[1:]
        if self.children:
            self.children[0].lstrip()

    def rstrip(self):
        while self.children and self.children[-1].is_whitespace():
            self.children = self.children[:-1]
        if self.children:
            self.children[-1].rstrip()

    def collapse_whitespace(self):
        """Remove collapsible white-space from HTML.

        HTML in docstrings often contains extraneous white-space around tags,
        for readability. Browsers would collapse this white-space before
        rendering. If not removed before conversion to RST where white-space is
        part of the syntax, for example for indentation, it can result in
        incorrect output.
        """
        self.lstrip()
        self.rstrip()
        for child in self.children:
            child.collapse_whitespace()