import base64
import collections
import logging
import struct
import functools
from os_ken import exception
from os_ken import utils
from os_ken.lib import stringify
from os_ken.ofproto import ofproto_common
class _namedtuple(StringifyMixin, collections.namedtuple(typename, fields, **kwargs)):
    pass