from collections.abc import Mapping
import copy
import logging
import sys
import traceback
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import oslo_messaging
from oslo_messaging import _utils as utils
class RPCException(Exception):
    msg_fmt = 'An unknown RPC related exception occurred.'

    def __init__(self, message=None, **kwargs):
        self.kwargs = kwargs
        if not message:
            try:
                message = self.msg_fmt % kwargs
            except Exception:
                LOG.exception('Exception in string format operation, kwargs are:')
                for name, value in kwargs.items():
                    LOG.error('%s: %s', name, value)
                message = self.msg_fmt
        super(RPCException, self).__init__(message)