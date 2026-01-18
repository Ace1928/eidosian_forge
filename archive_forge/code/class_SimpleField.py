import xml.sax.saxutils
class SimpleField(XMLTemplate):
    """
    A Simple name/value pair that can be easily rendered as XML.

    >>> SimpleField('Text', 'A text string').get_as_xml()
    '<Text>A text string</Text>'
    """
    template = '<%(field)s>%(value)s</%(field)s>'

    def __init__(self, field, value):
        self.field = field
        self.value = value