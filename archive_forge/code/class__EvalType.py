import ast
import copy
import re
import flask
import jsonschema
from oslo_config import cfg
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
class _EvalType(object):
    """Mapping rule evaluation types."""
    ANY_ONE_OF = 'any_one_of'
    NOT_ANY_OF = 'not_any_of'
    BLACKLIST = 'blacklist'
    WHITELIST = 'whitelist'