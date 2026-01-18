import xml.sax.saxutils
class QuestionContent(OrderedContent):
    template = '<QuestionContent>%(content)s</QuestionContent>'

    def get_as_xml(self):
        content = super(QuestionContent, self).get_as_xml()
        return self.template % vars()