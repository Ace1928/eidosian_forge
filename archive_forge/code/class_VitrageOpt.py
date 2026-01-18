import os
import requests
from keystoneauth1 import loading
from keystoneauth1 import plugin
from oslo_log import log
class VitrageOpt(loading.Opt):

    @property
    def argparse_args(self):
        return ['--%s' % o.name for o in self._all_opts]

    @property
    def argparse_default(self):
        for o in self._all_opts:
            v = os.environ.get('VITRAGE_%s' % o.name.replace('-', '_').upper())
            if v:
                return v
        return self.default