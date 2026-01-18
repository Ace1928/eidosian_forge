import importlib
import logging
import sys
from osc_lib import clientmanager
from osc_lib import shell
import stevedore
def _fallback_load_auth_plugin(self, e):
    if self._cli_options.config['auth']['token'] == 'x':
        self._cli_options.config['auth_type'] = self._original_auth_type
        del self._cli_options.config['auth']['token']
        del self._cli_options.config['auth']['endpoint']
        self._cli_options._auth = self._cli_options._openstack_config.load_auth_plugin(self._cli_options.config)
    else:
        raise e