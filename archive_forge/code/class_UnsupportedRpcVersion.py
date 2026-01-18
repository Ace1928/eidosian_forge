from collections.abc import Mapping
import copy
import logging
import sys
import traceback
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import oslo_messaging
from oslo_messaging import _utils as utils
class UnsupportedRpcVersion(RPCException):
    msg_fmt = 'Specified RPC version, %(version)s, not supported by this endpoint.'