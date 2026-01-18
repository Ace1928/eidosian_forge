import xml.sax.saxutils
class ExternalQuestion(ValidatingXML):
    """
    An object for constructing an External Question.
    """
    schema_url = 'http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd'
    template = '<ExternalQuestion xmlns="%(schema_url)s"><ExternalURL>%%(external_url)s</ExternalURL><FrameHeight>%%(frame_height)s</FrameHeight></ExternalQuestion>' % vars()

    def __init__(self, external_url, frame_height):
        self.external_url = xml.sax.saxutils.escape(external_url)
        self.frame_height = frame_height

    def get_as_params(self, label='ExternalQuestion'):
        return {label: self.get_as_xml()}

    def get_as_xml(self):
        return self.template % vars(self)