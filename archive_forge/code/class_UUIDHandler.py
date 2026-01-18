import datetime
import functools
import itertools
import uuid
from xmlrpc import client as xmlrpclib
import msgpack
from oslo_utils import importutils
class UUIDHandler(object):
    identity = 0
    handles = (uuid.UUID,)

    @staticmethod
    def serialize(obj):
        return str(obj.hex).encode('ascii')

    @staticmethod
    def deserialize(data):
        return uuid.UUID(hex=str(data, encoding='ascii'))