from html5lib.treebuilders import _base, etree as etree_builders
from lxml import html, etree
class DocumentType:

    def __init__(self, name, publicId, systemId):
        self.name = name
        self.publicId = publicId
        self.systemId = systemId