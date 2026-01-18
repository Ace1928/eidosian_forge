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
def _evaluate_values_by_regex(self, values, assertion_values):
    return [assertion for assertion in assertion_values if any([re.search(regex, assertion) for regex in values])]