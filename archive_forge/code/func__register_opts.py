import os
from oslotest import base
from requests import HTTPError
import requests_mock
import testtools
from oslo_config import _list_opts
from oslo_config import cfg
from oslo_config import fixture
from oslo_config import sources
from oslo_config.sources import _uri
def _register_opts(self, opts):
    for g in opts.keys():
        for o, (t, _) in opts[g].items():
            self.conf.register_opt(t(o), g if g != 'DEFAULT' else None)