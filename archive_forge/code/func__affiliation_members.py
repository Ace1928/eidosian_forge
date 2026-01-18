import logging
from saml2.attribute_resolver import AttributeResolver
from saml2.saml import NAMEID_FORMAT_PERSISTENT
def _affiliation_members(self):
    """
        Get the member of the Virtual Organization from the metadata,
        more specifically from AffiliationDescriptor.
        """
    return self.sp.config.metadata.vo_members(self._name)