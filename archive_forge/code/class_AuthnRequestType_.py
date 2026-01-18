import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class AuthnRequestType_(RequestAbstractType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:AuthnRequestType element"""
    c_tag = 'AuthnRequestType'
    c_namespace = NAMESPACE
    c_children = RequestAbstractType_.c_children.copy()
    c_attributes = RequestAbstractType_.c_attributes.copy()
    c_child_order = RequestAbstractType_.c_child_order[:]
    c_cardinality = RequestAbstractType_.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}Subject'] = ('subject', saml.Subject)
    c_cardinality['subject'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:protocol}NameIDPolicy'] = ('name_id_policy', NameIDPolicy)
    c_cardinality['name_id_policy'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}Conditions'] = ('conditions', saml.Conditions)
    c_cardinality['conditions'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:protocol}RequestedAuthnContext'] = ('requested_authn_context', RequestedAuthnContext)
    c_cardinality['requested_authn_context'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:protocol}Scoping'] = ('scoping', Scoping)
    c_cardinality['scoping'] = {'min': 0, 'max': 1}
    c_attributes['ForceAuthn'] = ('force_authn', 'boolean', False)
    c_attributes['IsPassive'] = ('is_passive', 'boolean', False)
    c_attributes['ProtocolBinding'] = ('protocol_binding', 'anyURI', False)
    c_attributes['AssertionConsumerServiceIndex'] = ('assertion_consumer_service_index', 'unsignedShort', False)
    c_attributes['AssertionConsumerServiceURL'] = ('assertion_consumer_service_url', 'anyURI', False)
    c_attributes['AttributeConsumingServiceIndex'] = ('attribute_consuming_service_index', 'unsignedShort', False)
    c_attributes['ProviderName'] = ('provider_name', 'string', False)
    c_child_order.extend(['subject', 'name_id_policy', 'conditions', 'requested_authn_context', 'scoping'])

    def __init__(self, subject=None, name_id_policy=None, conditions=None, requested_authn_context=None, scoping=None, force_authn=None, is_passive=None, protocol_binding=None, assertion_consumer_service_index=None, assertion_consumer_service_url=None, attribute_consuming_service_index=None, provider_name=None, issuer=None, signature=None, extensions=None, id=None, version=None, issue_instant=None, destination=None, consent=None, text=None, extension_elements=None, extension_attributes=None):
        RequestAbstractType_.__init__(self, issuer=issuer, signature=signature, extensions=extensions, id=id, version=version, issue_instant=issue_instant, destination=destination, consent=consent, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.subject = subject
        self.name_id_policy = name_id_policy
        self.conditions = conditions
        self.requested_authn_context = requested_authn_context
        self.scoping = scoping
        self.force_authn = force_authn
        self.is_passive = is_passive
        self.protocol_binding = protocol_binding
        self.assertion_consumer_service_index = assertion_consumer_service_index
        self.assertion_consumer_service_url = assertion_consumer_service_url
        self.attribute_consuming_service_index = attribute_consuming_service_index
        self.provider_name = provider_name