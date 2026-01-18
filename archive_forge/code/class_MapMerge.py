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
class MapMerge(function.Function):
    """A function for merging maps.

    Takes the form::

        map_merge:
          - <k1>: <v1>
            <k2>: <v2>
          - <k1>: <v3>

    And resolves to::

        {"<k1>": "<v3>", "<k2>": "<v2>"}

    """

    def __init__(self, stack, fn_name, args):
        super(MapMerge, self).__init__(stack, fn_name, args)
        example = _('"%s" : [ { "key1": "val1" }, { "key2": "val2" } ]') % fn_name
        self.fmt_data = {'fn_name': fn_name, 'example': example}

    def result(self):
        args = function.resolve(self.args)
        if not isinstance(args, collections.abc.Sequence):
            raise TypeError(_('Incorrect arguments to "%(fn_name)s" should be: %(example)s') % self.fmt_data)

        def ensure_map(m):
            if m is None:
                return {}
            elif isinstance(m, collections.abc.Mapping):
                return m
            else:
                msg = _('Incorrect arguments: Items to merge must be maps. {} is type {} instead of a dict'.format(repr(m)[:200], type(m)))
                raise TypeError(msg)
        ret_map = {}
        for m in args:
            ret_map.update(ensure_map(m))
        return ret_map