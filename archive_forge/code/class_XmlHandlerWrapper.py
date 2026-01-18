import xml.sax
from boto.compat import StringIO
class XmlHandlerWrapper(object):

    def __init__(self, root_node, connection):
        self.handler = XmlHandler(root_node, connection)
        self.parser = xml.sax.make_parser()
        self.parser.setContentHandler(self.handler)
        self.parser.setFeature(xml.sax.handler.feature_external_ges, 0)

    def parseString(self, content):
        return self.parser.parse(StringIO(content))