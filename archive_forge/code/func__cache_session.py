import logging
from saml2.attribute_resolver import AttributeResolver
from saml2.saml import NAMEID_FORMAT_PERSISTENT
def _cache_session(self, session_info):
    return True