import xml.sax.saxutils
class OrderedContent(list):

    def append_field(self, field, value):
        self.append(SimpleField(field, value))

    def get_as_xml(self):
        return ''.join((item.get_as_xml() for item in self))