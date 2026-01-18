from decimal import Decimal
from boto.compat import filter, map
class ComplexWeight(ResponseElement):

    def __repr__(self):
        return '{0} {1}'.format(self.Value, self.Unit)

    def __float__(self):
        return float(self.Value)

    def __str__(self):
        return str(self.Value)

    @strip_namespace
    def startElement(self, name, attrs, connection):
        if name not in ('Unit', 'Value'):
            message = 'Unrecognized tag {0} in ComplexWeight'.format(name)
            raise AssertionError(message)
        return super(ComplexWeight, self).startElement(name, attrs, connection)

    @strip_namespace
    def endElement(self, name, value, connection):
        if name == 'Value':
            value = Decimal(value)
        super(ComplexWeight, self).endElement(name, value, connection)