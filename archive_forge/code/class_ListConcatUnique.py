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
class ListConcatUnique(ListConcat):
    """A function for extending lists with unique items.

    list_concat_unique is identical to the list_concat function, only
    contains unique items in retuning list.
    """
    _unique = True