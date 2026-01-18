import saml2
from saml2 import SamlBase
class RequestType_(SamlBase):
    """The urn:liberty:paos:2003-08:RequestType element"""
    c_tag = 'RequestType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['responseConsumerURL'] = ('response_consumer_url', 'anyURI', True)
    c_attributes['service'] = ('service', 'anyURI', True)
    c_attributes['messageID'] = ('message_id', 'None', False)
    c_attributes['{http://schemas.xmlsoap.org/soap/envelope/}mustUnderstand'] = ('must_understand', 'None', True)
    c_attributes['{http://schemas.xmlsoap.org/soap/envelope/}actor'] = ('actor', 'None', True)

    def __init__(self, response_consumer_url=None, service=None, message_id=None, must_understand=None, actor=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.response_consumer_url = response_consumer_url
        self.service = service
        self.message_id = message_id
        self.must_understand = must_understand
        self.actor = actor