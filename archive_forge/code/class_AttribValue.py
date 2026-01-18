from io import StringIO
class AttribValue:

    def __init__(self, attribname):
        self.attribname = attribname
        if self.attribname == 'xmlns':
            self.value = self.value_ns

    def value_ns(self, elem):
        return elem.uri

    def value(self, elem):
        if self.attribname in elem.attributes:
            return elem.attributes[self.attribname]
        else:
            return None