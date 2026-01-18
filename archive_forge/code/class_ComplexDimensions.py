from decimal import Decimal
from boto.compat import filter, map
class ComplexDimensions(ResponseElement):
    _dimensions = ('Height', 'Length', 'Width', 'Weight')

    def __repr__(self):
        values = [getattr(self, key, None) for key in self._dimensions]
        values = filter(None, values)
        return 'x'.join(map('{0.Value:0.2f}{0[Units]}'.format, values))

    @strip_namespace
    def startElement(self, name, attrs, connection):
        if name not in self._dimensions:
            message = 'Unrecognized tag {0} in ComplexDimensions'.format(name)
            raise AssertionError(message)
        setattr(self, name, Dimension(attrs.copy()))

    @strip_namespace
    def endElement(self, name, value, connection):
        if name in self._dimensions:
            value = Decimal(value or '0')
        ResponseElement.endElement(self, name, value, connection)