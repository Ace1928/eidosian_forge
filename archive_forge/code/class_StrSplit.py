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
class StrSplit(function.Function):
    """A function for splitting delimited strings into a list.

    Optionally extracting a specific list member by index.

    Takes the form::

        str_split:
          - <delimiter>
          - <string>
          - <index>

    If <index> is specified, the specified list item will be returned
    otherwise, the whole list is returned, similar to get_attr with
    path based attributes accessing lists.
    """

    def __init__(self, stack, fn_name, args):
        super(StrSplit, self).__init__(stack, fn_name, args)
        example = '"%s" : [ ",", "apples,pears", <index>]' % fn_name
        self.fmt_data = {'fn_name': fn_name, 'example': example}
        self.fn_name = fn_name
        if isinstance(args, (str, collections.abc.Mapping)):
            raise TypeError(_('Incorrect arguments to "%(fn_name)s" should be: %(example)s') % self.fmt_data)

    def result(self):
        args = function.resolve(self.args)
        try:
            delim = args.pop(0)
            str_to_split = args.pop(0)
        except (AttributeError, IndexError):
            raise ValueError(_('Incorrect arguments to "%(fn_name)s" should be: %(example)s') % self.fmt_data)
        if str_to_split is None:
            return None
        split_list = str_to_split.split(delim)
        if args:
            try:
                index = int(args.pop(0))
            except ValueError:
                raise ValueError(_('Incorrect index to "%(fn_name)s" should be: %(example)s') % self.fmt_data)
            else:
                try:
                    res = split_list[index]
                except IndexError:
                    raise ValueError(_('Incorrect index to "%(fn_name)s" should be between 0 and %(max_index)s') % {'fn_name': self.fn_name, 'max_index': len(split_list) - 1})
        else:
            res = split_list
        return res