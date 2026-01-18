from abc import ABCMeta
from abc import abstractmethod
import logging
import sys
import threading
from oslo_config import cfg
from oslo_utils import eventletutils
from oslo_messaging import _utils as utils
from oslo_messaging import dispatcher
from oslo_messaging import serializer as msg_serializer
from oslo_messaging import server as msg_server
from oslo_messaging import target as msg_target
class LegacyRPCAccessPolicy(RPCAccessPolicyBase):
    """The legacy access policy allows RPC access to all callable endpoint
    methods including private methods (methods prefixed by '_')
    """

    def is_allowed(self, endpoint, method):
        return True