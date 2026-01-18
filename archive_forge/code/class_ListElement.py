class ListElement(list):
    """
    A :py:class:`list` subclass that has some additional methods
    for interacting with Amazon's XML API.
    """

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'member':
            self.append(value)