import datetime
import itertools
from xmlrpc import client as xmlrpclib
import netaddr
from oslotest import base as test_base
from oslo_serialization import msgpackutils
from oslo_utils import uuidutils
class MySpecialSetHandler(object):
    handles = (set,)
    identity = msgpackutils.SetHandler.identity