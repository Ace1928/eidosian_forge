class _XMLKeyValue(object):

    def __init__(self, translator, container=None):
        self.translator = translator
        if container:
            self.container = container
        else:
            self.container = self

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        for xml_key, attr_name in self.translator:
            if name == xml_key:
                setattr(self.container, attr_name, value)

    def to_xml(self):
        parts = []
        for xml_key, attr_name in self.translator:
            content = getattr(self.container, attr_name)
            if content is not None:
                parts.append(tag(xml_key, content))
        return ''.join(parts)