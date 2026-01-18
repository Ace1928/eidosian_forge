from html.parser import HTMLParser
from itertools import zip_longest
def _has_nested_tags(self):
    return any((isinstance(child, TagNode) for child in self.children))