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
class RepeatWithNestedLoop(RepeatWithMap):
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

    This function also allows to specify 'permutations' to decide
    whether to iterate nested the over all the permutations of the
    elements in the given lists.

    Takes the form::

        repeat:
          template:
            var: %var%
            bar: %bar%
          for_each:
            %var%: <list1>
            %bar%: <list2>
          permutations: false

    If 'permutations' is not specified, we set the default value to true to
    compatible with before behavior. The args have to be lists instead of
    dicts if 'permutations' is False because keys in a dict are unordered,
    and the list args all have to be of the same length.
    """

    def _parse_args(self):
        super(RepeatWithNestedLoop, self)._parse_args()
        self._nested_loop = self.args.get('permutations', True)
        if not isinstance(self._nested_loop, bool):
            raise TypeError(_('"permutations" should be boolean type for %s function.') % self.fn_name)

    def _valid_arg(self, arg):
        if self._nested_loop:
            super(RepeatWithNestedLoop, self)._valid_arg(arg)
        else:
            Repeat._valid_arg(self, arg)