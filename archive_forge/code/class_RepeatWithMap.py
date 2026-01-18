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
class RepeatWithMap(Repeat):
    """A function for iterating over a list of items or a dict of keys.

    Takes the form::

        repeat:
            template:
                <body>
            for_each:
                <var>: <list> or <dict>

    The result is a new list of the same size as <list> or <dict>, where each
    element is a copy of <body> with any occurrences of <var> replaced with the
    corresponding item of <list> or key of <dict>.
    """

    def _valid_arg(self, arg):
        if not (isinstance(arg, (collections.abc.Sequence, collections.abc.Mapping, function.Function)) and (not isinstance(arg, str))):
            raise TypeError(_('The values of the "for_each" argument to "%s" must be lists or maps') % self.fn_name)