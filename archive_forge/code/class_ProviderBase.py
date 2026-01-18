from __future__ import absolute_import, division, print_function
import json
from threading import RLock
from ansible.module_utils.six import itervalues
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
class ProviderBase(object):
    supported_connections = ()

    def __init__(self, params, connection=None, check_mode=False):
        self.params = params
        self.connection = connection
        self.check_mode = check_mode

    @property
    def capabilities(self):
        if not hasattr(self, '_capabilities'):
            resp = self.from_json(self.connection.get_capabilities())
            setattr(self, '_capabilities', resp)
        return getattr(self, '_capabilities')

    def get_value(self, path):
        params = self.params.copy()
        for key in path.split('.'):
            params = params[key]
        return params

    def get_facts(self, subset=None):
        raise NotImplementedError(self.__class__.__name__)

    def edit_config(self):
        raise NotImplementedError(self.__class__.__name__)