import datetime
import functools
import itertools
import uuid
from xmlrpc import client as xmlrpclib
import msgpack
from oslo_utils import importutils
class NetAddrIPHandler(object):
    identity = 3
    handles = (netaddr.IPAddress,)

    @staticmethod
    def serialize(obj):
        return msgpack.packb(obj.value)

    @staticmethod
    def deserialize(data):
        return netaddr.IPAddress(msgpack.unpackb(data))