import saml2
from saml2 import SamlBase
class PolicyAttachment(SamlBase):
    """The http://schemas.xmlsoap.org/ws/2004/09/policy:PolicyAttachment element"""
    c_tag = 'PolicyAttachment'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://schemas.xmlsoap.org/ws/2004/09/policy}AppliesTo'] = ('applies_to', AppliesTo)
    c_cardinality['policy'] = {'min': 0}
    c_children['{http://schemas.xmlsoap.org/ws/2004/09/policy}PolicyReference'] = ('policy_reference', [PolicyReference])
    c_cardinality['policy_reference'] = {'min': 0}
    c_child_order.extend(['applies_to', 'policy', 'policy_reference'])

    def __init__(self, applies_to=None, policy=None, policy_reference=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.applies_to = applies_to
        self.policy = policy or []
        self.policy_reference = policy_reference or []