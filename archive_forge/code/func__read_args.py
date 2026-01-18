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
def _read_args(self):
    if not 2 <= len(self.args) <= 3:
        raise ValueError()
    if len(self.args) == 2:
        condition, value_if_true = self.args
        return (condition, value_if_true, Ellipsis)
    return self.args