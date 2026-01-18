import fixtures
from oslo_config import cfg
def _unregister_config_opts(self):
    for group in self._registered_config_opts:
        self.conf.unregister_opts(self._registered_config_opts[group], group=group)