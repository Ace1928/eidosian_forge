import xml.sax.saxutils
class FormattedContent(XMLTemplate):
    schema_url = 'http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/FormattedContentXHTMLSubset.xsd'
    template = '<FormattedContent><![CDATA[%(content)s]]></FormattedContent>'

    def __init__(self, content):
        self.content = content