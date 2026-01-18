from collections.abc import Mapping
import copy
import logging
import sys
import traceback
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import oslo_messaging
from oslo_messaging import _utils as utils
class DuplicateMessageError(RPCException):
    msg_fmt = 'Found duplicate message(%(msg_id)s). Skipping it.'