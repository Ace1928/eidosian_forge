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
def _validate_replacement(self, value, param):

    def _raise_empty_param_value_error():
        raise ValueError(_('%(name)s has an undefined or empty value for param %(param)s, must be a defined non-empty value') % {'name': self.fn_name, 'param': param})
    if value is None:
        if self._allow_empty_value:
            return ''
        else:
            _raise_empty_param_value_error()
    if not isinstance(value, (str, int, float, bool)):
        if isinstance(value, (collections.abc.Mapping, collections.abc.Sequence)):
            if not self._allow_empty_value and len(value) == 0:
                _raise_empty_param_value_error()
            try:
                return jsonutils.dumps(value, default=None, sort_keys=True)
            except TypeError:
                raise TypeError(_('"%(name)s" params must be strings, numbers, list or map. Failed to json serialize %(value)s') % {'name': self.fn_name, 'value': value})
        else:
            raise TypeError(_('"%s" params must be strings, numbers, list or map.') % self.fn_name)
    ret_value = str(value)
    if not self._allow_empty_value and (not ret_value):
        _raise_empty_param_value_error()
    return ret_value