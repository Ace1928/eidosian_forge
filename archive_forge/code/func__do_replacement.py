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
def _do_replacement(self, keys, values, template):
    if isinstance(template, str):
        for key, value in zip(keys, values):
            template = template.replace(key, value)
        return template
    elif isinstance(template, collections.abc.Sequence):
        return [self._do_replacement(keys, values, elem) for elem in template]
    elif isinstance(template, collections.abc.Mapping):
        return dict(((self._do_replacement(keys, values, k), self._do_replacement(keys, values, v)) for k, v in template.items()))
    else:
        return template