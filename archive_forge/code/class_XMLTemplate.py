import xml.sax.saxutils
class XMLTemplate(object):

    def get_as_xml(self):
        return self.template % vars(self)