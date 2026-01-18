import abc
from oslo_log import log as logging
from oslo_utils import importutils
from os_brick import exception
from os_brick.i18n import _
class CacheEngineBase(object, metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        self._root_helper = kwargs.get('root_helper')

    @abc.abstractmethod
    def is_engine_ready(self, **kwargs):
        return

    @abc.abstractmethod
    def attach_volume(self, **kwargs):
        return

    @abc.abstractmethod
    def detach_volume(self, **kwargs):
        return