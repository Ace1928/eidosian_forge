import xml.sax.saxutils
class AnswerSpecification(object):
    template = '<AnswerSpecification>%(spec)s</AnswerSpecification>'

    def __init__(self, spec):
        self.spec = spec

    def get_as_xml(self):
        spec = self.spec.get_as_xml()
        return self.template % vars()