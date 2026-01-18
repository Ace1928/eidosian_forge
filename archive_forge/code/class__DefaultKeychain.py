from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import base64
import errno
import io
import json
import logging
import os
import subprocess
from containerregistry.client import docker_name
import httplib2
from oauth2client import client as oauth2client
import six
class _DefaultKeychain(Keychain):
    """This implements the default docker credential resolution."""

    def __init__(self):
        self._config_dir = None
        self._config_file = 'config.json'

    def setCustomConfigDir(self, config_dir):
        if not os.path.isdir(config_dir):
            raise Exception('Attempting to override docker configuration directory to invalid directory: {}'.format(config_dir))
        self._config_dir = config_dir

    def Resolve(self, name):
        logging.info('Loading Docker credentials for repository %r', str(name))
        config_file = None
        if self._config_dir is not None:
            config_file = os.path.join(self._config_dir, self._config_file)
        else:
            config_file = os.path.join(_GetConfigDirectory(), self._config_file)
        try:
            with io.open(config_file, u'r', encoding='utf8') as reader:
                cfg = json.loads(reader.read())
        except IOError:
            return Anonymous()
        cred_store = cfg.get('credHelpers', {})
        for form in _FORMATS:
            if form % name.registry in cred_store:
                return Helper(cred_store[form % name.registry], name)
        if 'credsStore' in cfg:
            return Helper(cfg['credsStore'], name)
        auths = cfg.get('auths', {})
        for form in _FORMATS:
            if form % name.registry in auths:
                entry = auths[form % name.registry]
                if 'auth' in entry:
                    decoded = base64.b64decode(entry['auth']).decode('utf8')
                    username, password = decoded.split(':', 1)
                    return Basic(username, password)
                elif 'username' in entry and 'password' in entry:
                    return Basic(entry['username'], entry['password'])
                else:
                    raise Exception('Unsupported entry in "auth" section of Docker config: ' + json.dumps(entry))
        return Anonymous()