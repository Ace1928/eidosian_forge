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
def _attr_path(self):
    if self._attribute is None:
        return attributes.ALL_ATTRIBUTES
    return super(GetAttAllAttributes, self)._attr_path()