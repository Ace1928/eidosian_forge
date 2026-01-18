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
@staticmethod
def _ast_literal_eval(value):
    try:
        values = ast.literal_eval(value)
        if not isinstance(values, list):
            raise ValueError
    except (ValueError, SyntaxError):
        values = [value]
    return values