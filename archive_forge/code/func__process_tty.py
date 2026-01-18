from urllib import parse
from zunclient import api_versions
from zunclient.common import base
from zunclient.common import utils
from zunclient import exceptions
def _process_tty(self, kwargs):
    tty_microversion = api_versions.APIVersion('1.36')
    if self.api_version >= tty_microversion:
        if 'interactive' in kwargs and 'tty' not in kwargs:
            kwargs['tty'] = kwargs['interactive']