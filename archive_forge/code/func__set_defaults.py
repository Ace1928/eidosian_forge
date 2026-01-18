import copy
import urllib
from oslo_serialization import jsonutils
from keystoneauth1 import exceptions
from mistralclient import utils
def _set_defaults(self):
    for k, v in self.defaults.items():
        if k not in self._data:
            self._data[k] = v