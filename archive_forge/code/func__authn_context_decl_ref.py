import copy
import importlib
import logging
import re
from warnings import warn as _warn
from saml2 import saml
from saml2 import xmlenc
from saml2.attribute_converter import ac_factory
from saml2.attribute_converter import from_local
from saml2.attribute_converter import get_local_name
from saml2.s_utils import MissingValue
from saml2.s_utils import assertion_factory
from saml2.s_utils import factory
from saml2.s_utils import sid
from saml2.saml import NAME_FORMAT_URI
from saml2.time_util import in_a_while
from saml2.time_util import instant
def _authn_context_decl_ref(decl_ref, authn_auth=None):
    """
    Construct the authn context with a authn context declaration reference
    :param decl_ref: The authn context declaration reference
    :param authn_auth: Authenticating Authority
    :return: An AuthnContext instance
    """
    return factory(saml.AuthnContext, authn_context_decl_ref=decl_ref, authenticating_authority=factory(saml.AuthenticatingAuthority, text=authn_auth))