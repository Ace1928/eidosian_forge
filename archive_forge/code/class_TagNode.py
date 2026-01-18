from html.parser import HTMLParser
from itertools import zip_longest
class TagNode(StemNode):
    """
    A generic Tag node. It will verify that handlers exist before writing.
    """

    def __init__(self, tag, attrs=None, parent=None):
        super().__init__(parent)
        self.attrs = attrs
        self.tag = tag

    def _has_nested_tags(self):
        return any((isinstance(child, TagNode) for child in self.children))

    def write(self, doc, next_child=None):
        prioritize_nested_tags = self.tag in OMIT_SELF_TAGS and self._has_nested_tags()
        prioritize_parent_tag = isinstance(self.parent, TagNode) and self.parent.tag in PRIORITY_PARENT_TAGS and (self.tag in OMIT_NESTED_TAGS)
        if prioritize_nested_tags or prioritize_parent_tag:
            self._write_children(doc)
            return
        self._write_start(doc)
        self._write_children(doc)
        self._write_end(doc, next_child)

    def collapse_whitespace(self):
        """Remove collapsible white-space.

        All tags collapse internal whitespace. Block-display HTML tags also
        strip all leading and trailing whitespace.

        Approximately follows the specification used in browsers:
        https://www.w3.org/TR/css-text-3/#white-space-rules
        https://developer.mozilla.org/en-US/docs/Web/API/Document_Object_Model/Whitespace
        """
        if self.tag in HTML_BLOCK_DISPLAY_TAGS:
            self.lstrip()
            self.rstrip()
        for prev, cur in zip(self.children[:-1], self.children[1:]):
            if isinstance(prev, DataNode) and prev.endswith_whitespace() and cur.startswith_whitespace():
                cur.lstrip()
        for cur, nxt in zip(self.children[:-1], self.children[1:]):
            if isinstance(nxt, DataNode) and cur.endswith_whitespace() and nxt.startswith_whitespace():
                cur.rstrip()
        for child in self.children:
            child.collapse_whitespace()

    def _write_start(self, doc):
        handler_name = 'start_%s' % self.tag
        if hasattr(doc.style, handler_name):
            getattr(doc.style, handler_name)(self.attrs)

    def _write_end(self, doc, next_child):
        handler_name = 'end_%s' % self.tag
        if hasattr(doc.style, handler_name):
            if handler_name == 'end_a':
                getattr(doc.style, handler_name)(next_child)
            else:
                getattr(doc.style, handler_name)()