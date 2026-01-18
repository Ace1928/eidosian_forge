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
def _do_dispatch(self, endpoint, method, ctxt, args):
    ctxt = self.serializer.deserialize_context(ctxt)
    new_args = dict()
    for argname, arg in args.items():
        new_args[argname] = self.serializer.deserialize_entity(ctxt, arg)
    func = getattr(endpoint, method)
    result = func(ctxt, **new_args)
    return self.serializer.serialize_entity(ctxt, result)