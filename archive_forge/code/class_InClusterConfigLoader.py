import os
from kubernetes.client import Configuration
from .config_exception import ConfigException
class InClusterConfigLoader(object):

    def __init__(self, token_filename, cert_filename, environ=os.environ):
        self._token_filename = token_filename
        self._cert_filename = cert_filename
        self._environ = environ

    def load_and_set(self):
        self._load_config()
        self._set_config()

    def _load_config(self):
        if SERVICE_HOST_ENV_NAME not in self._environ or SERVICE_PORT_ENV_NAME not in self._environ:
            raise ConfigException('Service host/port is not set.')
        if not self._environ[SERVICE_HOST_ENV_NAME] or not self._environ[SERVICE_PORT_ENV_NAME]:
            raise ConfigException('Service host/port is set but empty.')
        self.host = 'https://' + _join_host_port(self._environ[SERVICE_HOST_ENV_NAME], self._environ[SERVICE_PORT_ENV_NAME])
        if not os.path.isfile(self._token_filename):
            raise ConfigException('Service token file does not exists.')
        with open(self._token_filename) as f:
            self.token = f.read()
            if not self.token:
                raise ConfigException('Token file exists but empty.')
        if not os.path.isfile(self._cert_filename):
            raise ConfigException('Service certification file does not exists.')
        with open(self._cert_filename) as f:
            if not f.read():
                raise ConfigException('Cert file exists but empty.')
        self.ssl_ca_cert = self._cert_filename

    def _set_config(self):
        configuration = Configuration()
        configuration.host = self.host
        configuration.ssl_ca_cert = self.ssl_ca_cert
        configuration.api_key['authorization'] = 'bearer ' + self.token
        Configuration.set_default(configuration)