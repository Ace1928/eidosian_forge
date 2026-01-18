from lxml import etree
from testtools import matchers
class XMLMismatch(matchers.Mismatch):

    def __init__(self, expected, other):
        self.expected = expected
        self.other = other

    def describe(self):

        def pretty_xml(xml):
            parser = etree.XMLParser(remove_blank_text=True)
            doc = etree.fromstring(xml.strip(), parser)
            return etree.tostring(doc, encoding='utf-8', pretty_print=True).decode('utf-8')
        return 'expected =\n%s\nactual =\n%s' % (pretty_xml(self.expected), pretty_xml(self.other))