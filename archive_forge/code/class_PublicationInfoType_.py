import saml2
from saml2 import SamlBase
from saml2 import md
class PublicationInfoType_(SamlBase):
    """The urn:oasis:names:tc:SAML:metadata:rpi:PublicationInfoType element"""
    c_tag = 'PublicationInfoType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:metadata:rpi}UsagePolicy'] = ('usage_policy', [UsagePolicy])
    c_cardinality['usage_policy'] = {'min': 0}
    c_attributes['publisher'] = ('publisher', 'string', True)
    c_attributes['creationInstant'] = ('creation_instant', 'dateTime', False)
    c_attributes['publicationId'] = ('publication_id', 'string', False)
    c_child_order.extend(['usage_policy'])

    def __init__(self, usage_policy=None, publisher=None, creation_instant=None, publication_id=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.usage_policy = usage_policy or []
        self.publisher = publisher
        self.creation_instant = creation_instant
        self.publication_id = publication_id