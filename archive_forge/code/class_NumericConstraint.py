import xml.sax.saxutils
class NumericConstraint(Constraint):
    attribute_names = ('minValue', 'maxValue')
    template = '<IsNumeric %(attrs)s />'

    def __init__(self, min_value=None, max_value=None):
        self.attribute_values = (min_value, max_value)