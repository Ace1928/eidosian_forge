from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
class _Loader:

    def get_app(self, name=None, global_conf=None):
        return self.app_context(name=name, global_conf=global_conf).create()

    def get_filter(self, name=None, global_conf=None):
        return self.filter_context(name=name, global_conf=global_conf).create()

    def get_server(self, name=None, global_conf=None):
        return self.server_context(name=name, global_conf=global_conf).create()

    def app_context(self, name=None, global_conf=None):
        return self.get_context(APP, name=name, global_conf=global_conf)

    def filter_context(self, name=None, global_conf=None):
        return self.get_context(FILTER, name=name, global_conf=global_conf)

    def server_context(self, name=None, global_conf=None):
        return self.get_context(SERVER, name=name, global_conf=global_conf)
    _absolute_re = re.compile('^[a-zA-Z]+:')

    def absolute_name(self, name):
        """
        Returns true if the name includes a scheme
        """
        if name is None:
            return False
        return self._absolute_re.search(name)