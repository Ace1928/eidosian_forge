from html.parser import HTMLParser
from itertools import zip_longest
class HTMLTree:
    """
    A tree which handles HTML nodes. Designed to work with a python HTML parser,
    meaning that the current_node will be the most recently opened tag. When
    a tag is closed, the current_node moves up to the parent node.
    """

    def __init__(self, doc):
        self.doc = doc
        self.head = StemNode()
        self.current_node = self.head
        self.unhandled_tags = []

    def add_tag(self, tag, attrs=None, is_start=True):
        if not self._doc_has_handler(tag, is_start):
            self.unhandled_tags.append(tag)
            return
        if is_start:
            node = TagNode(tag, attrs)
            self.current_node.add_child(node)
            self.current_node = node
        else:
            self.current_node = self.current_node.parent

    def _doc_has_handler(self, tag, is_start):
        if is_start:
            handler_name = 'start_%s' % tag
        else:
            handler_name = 'end_%s' % tag
        return hasattr(self.doc.style, handler_name)

    def add_data(self, data):
        self.current_node.add_child(DataNode(data))

    def write(self):
        self.head.write(self.doc)