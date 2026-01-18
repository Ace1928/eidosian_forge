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
def _match_attr_name(attr, ava):
    name = attr['name'].lower()
    name_format = attr.get('name_format')
    friendly_name = attr.get('friendly_name')
    local_name = get_local_name(acs, name, name_format) or friendly_name or ''
    _fn = _match(local_name, ava) or _match(name, ava)
    return _fn