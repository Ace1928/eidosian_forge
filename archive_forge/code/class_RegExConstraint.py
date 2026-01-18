import xml.sax.saxutils
class RegExConstraint(Constraint):
    attribute_names = ('regex', 'errorText', 'flags')
    template = '<AnswerFormatRegex %(attrs)s />'

    def __init__(self, pattern, error_text=None, flags=None):
        self.attribute_values = (pattern, error_text, flags)

    def get_attributes(self):
        pairs = zip(self.attribute_names, self.attribute_values)
        attrs = ' '.join(('%s="%s"' % (name, value) for name, value in pairs if value is not None))
        return attrs