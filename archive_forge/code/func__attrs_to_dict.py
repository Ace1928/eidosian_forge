from xml.parsers import expat
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
def _attrs_to_dict(self, attrs):
    if isinstance(attrs, dict):
        return attrs
    return self.dict_constructor(zip(attrs[0::2], attrs[1::2]))