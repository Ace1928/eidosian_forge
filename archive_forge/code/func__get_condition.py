import collections
import functools
import hashlib
import itertools
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from urllib import parse as urlparse
import yaql
from yaql.language import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import function
def _get_condition(self, arg):
    if isinstance(arg, bool):
        return arg
    conditions = self.stack.t.conditions(self.stack)
    return conditions.is_enabled(arg)