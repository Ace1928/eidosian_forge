from lxml import etree
from testtools import matchers
def _sorted_children(doc):
    return sorted(doc.getchildren(), key=lambda el: el.tag)