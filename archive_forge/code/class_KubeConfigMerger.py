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
class KubeConfigMerger:
    """Reads and merges configuration from one or more kube-config's.

    The propery `config` can be passed to the KubeConfigLoader as config_dict.

    It uses a path attribute from ConfigNode to store the path to kubeconfig.
    This path is required to load certs from relative paths.

    A method `save_changes` updates changed kubeconfig's (it compares current
    state of dicts with).
    """

    def __init__(self, paths):
        self.paths = []
        self.config_files = {}
        self.config_merged = None
        for path in paths.split(ENV_KUBECONFIG_PATH_SEPARATOR):
            if path:
                path = os.path.expanduser(path)
                if os.path.exists(path):
                    self.paths.append(path)
                    self.load_config(path)
        self.config_saved = copy.deepcopy(self.config_files)

    @property
    def config(self):
        return self.config_merged

    def load_config(self, path):
        with open(path) as f:
            config = yaml.safe_load(f)
        if self.config_merged is None:
            config_merged = copy.deepcopy(config)
            for item in ('clusters', 'contexts', 'users'):
                config_merged[item] = []
            self.config_merged = ConfigNode(path, config_merged, path)
        for item in ('clusters', 'contexts', 'users'):
            self._merge(item, config[item], path)
        self.config_files[path] = config

    def _merge(self, item, add_cfg, path):
        for new_item in add_cfg:
            for exists in self.config_merged.value[item]:
                if exists['name'] == new_item['name']:
                    break
            else:
                self.config_merged.value[item].append(ConfigNode('{}/{}'.format(path, new_item), new_item, path))

    def save_changes(self):
        for path in self.paths:
            if self.config_saved[path] != self.config_files[path]:
                self.save_config(path)
        self.config_saved = copy.deepcopy(self.config_files)

    def save_config(self, path):
        with open(path, 'w') as f:
            yaml.safe_dump(self.config_files[path], f, default_flow_style=False)