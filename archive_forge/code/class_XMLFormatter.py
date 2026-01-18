from bs4.dammit import EntitySubstitution
class XMLFormatter(Formatter):
    """A generic Formatter for XML."""
    REGISTRY = {}

    def __init__(self, *args, **kwargs):
        super(XMLFormatter, self).__init__(self.XML, *args, **kwargs)