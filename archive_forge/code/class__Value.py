import abc
import base64
import collections
import copy
import functools
import io
import itertools
import math
import operator
import re
import socket
import struct
import netaddr
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib.packet import afi as addr_family
from os_ken.lib.packet import safi as subaddr_family
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import stream_parser
from os_ken.lib.packet import vxlan
from os_ken.lib.packet import mpls
from os_ken.lib import addrconv
from os_ken.lib import type_desc
from os_ken.lib.type_desc import TypeDisp
from os_ken.lib import ip
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.utils import binary_str
from os_ken.utils import import_module
class _Value(object):
    _VALUE_PACK_STR = None
    _VALUE_FIELDS = ['value']

    @staticmethod
    def do_init(cls_type, self, kwargs, **extra_kwargs):
        ourfields = {}
        for f in cls_type._VALUE_FIELDS:
            v = kwargs[f]
            del kwargs[f]
            ourfields[f] = v
        kwargs.update(extra_kwargs)
        super(cls_type, self).__init__(**kwargs)
        self.__dict__.update(ourfields)

    @classmethod
    def parse_value(cls, buf):
        values = struct.unpack_from(cls._VALUE_PACK_STR, bytes(buf))
        return dict(zip(cls._VALUE_FIELDS, values))

    def serialize_value(self):
        args = []
        for f in self._VALUE_FIELDS:
            args.append(getattr(self, f))
        return struct.pack(self._VALUE_PACK_STR, *args)