import atexit
import base64
import copy
import datetime
import json
import logging
import os
import platform
import tempfile
import time
import google.auth
import google.auth.transport.requests
import oauthlib.oauth2
import urllib3
from ruamel import yaml
from requests_oauthlib import OAuth2Session
from six import PY3
from kubernetes.client import ApiClient, Configuration
from kubernetes.config.exec_provider import ExecProvider
from .config_exception import ConfigException
from .dateutil import UTC, format_rfc3339, parse_rfc3339
class ConfigNode(object):
    """Remembers each config key's path and construct a relevant exception

    message in case of missing keys. The assumption is all access keys are
    present in a well-formed kube-config.
  """

    def __init__(self, name, value, path=None):
        self.name = name
        self.value = value
        self.path = path

    def __contains__(self, key):
        return key in self.value

    def __len__(self):
        return len(self.value)

    def safe_get(self, key):
        if isinstance(self.value, list) and isinstance(key, int) or key in self.value:
            return self.value[key]

    def __getitem__(self, key):
        v = self.safe_get(key)
        if not v:
            raise ConfigException('Invalid kube-config file. Expected key %s in %s' % (key, self.name))
        if isinstance(v, dict) or isinstance(v, list):
            return ConfigNode('%s/%s' % (self.name, key), v, self.path)
        else:
            return v

    def get_with_name(self, name, safe=False):
        if not isinstance(self.value, list):
            raise ConfigException('Invalid kube-config file. Expected %s to be a list' % self.name)
        result = None
        for v in self.value:
            if 'name' not in v:
                raise ConfigException("Invalid kube-config file. Expected all values in %s list to have 'name' key" % self.name)
            if v['name'] == name:
                if result is None:
                    result = v
                else:
                    raise ConfigException('Invalid kube-config file. Expected only one object with name %s in %s list' % (name, self.name))
        if result is not None:
            if isinstance(result, ConfigNode):
                return result
            else:
                return ConfigNode('%s[name=%s]' % (self.name, name), result, self.path)
        if safe:
            return None
        raise ConfigException('Invalid kube-config file. Expected object with name %s in %s list' % (name, self.name))