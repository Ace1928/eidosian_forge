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
def _authn_context_class_ref(authn_class, authn_auth=None):
    """
    Construct the authn context with a authn context class reference
    :param authn_class: The authn context class reference
    :param authn_auth: Authenticating Authority
    :return: An AuthnContext instance
    """
    cntx_class = factory(saml.AuthnContextClassRef, text=authn_class)
    if authn_auth:
        return factory(saml.AuthnContext, authn_context_class_ref=cntx_class, authenticating_authority=factory(saml.AuthenticatingAuthority, text=authn_auth))
    else:
        return factory(saml.AuthnContext, authn_context_class_ref=cntx_class)