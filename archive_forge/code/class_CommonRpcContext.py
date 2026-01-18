from collections.abc import Mapping
import copy
import logging
import sys
import traceback
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import oslo_messaging
from oslo_messaging import _utils as utils
class CommonRpcContext(object):

    def __init__(self, **kwargs):
        self.values = kwargs

    def __getattr__(self, key):
        try:
            return self.values[key]
        except KeyError:
            raise AttributeError(key)

    def to_dict(self):
        return copy.deepcopy(self.values)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def deepcopy(self):
        return self.from_dict(self.to_dict())

    def update_store(self):
        pass