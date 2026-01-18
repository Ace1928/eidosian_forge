import weakref
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import importutils
from stevedore import enabled
from heat.common import exception
from heat.common.i18n import _
from heat.common import pluginutils
class OpenStackClients(object):
    """Convenience class to create and cache client instances."""

    def __init__(self, context):
        self._context = weakref.ref(context)
        self._clients = {}
        self._client_plugins = {}

    @property
    def context(self):
        ctxt = self._context()
        assert ctxt is not None, 'Need a reference to the context'
        return ctxt

    def client_plugin(self, name):
        global _mgr
        if name in self._client_plugins:
            return self._client_plugins[name]
        if _mgr and name in _mgr.names():
            client_plugin = _mgr[name].plugin(self.context)
            self._client_plugins[name] = client_plugin
            return client_plugin

    def client(self, name, version=None):
        client_plugin = self.client_plugin(name)
        if client_plugin:
            if version:
                return client_plugin.client(version=version)
            else:
                return client_plugin.client()
        if name in self._clients:
            return self._clients[name]
        method_name = '_%s' % name
        if callable(getattr(self, method_name, None)):
            client = getattr(self, method_name)()
            self._clients[name] = client
            return client
        LOG.warning('Requested client "%s" not found', name)