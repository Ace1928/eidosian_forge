import requests
import tempfile
from oslo_config import cfg
from oslo_config import sources
class URIConfigurationSourceDriver(sources.ConfigurationSourceDriver):
    """A backend driver for remote files served through http[s].

    Required options:
      - uri: URI containing the file location.

    Non-required options:
      - ca_path: The path to a CA_BUNDLE file or directory with
                 certificates of trusted CAs.

      - client_cert: Client side certificate, as a single file path
                     containing either the certificate only or the
                     private key and the certificate.

      - client_key: Client side private key, in case client_cert is
                    specified but does not includes the private key.
    """
    _uri_driver_opts = [cfg.URIOpt('uri', schemes=['http', 'https'], required=True, sample_default='https://example.com/my-configuration.ini', help="Required option with the URI of the extra configuration file's location."), cfg.StrOpt('ca_path', sample_default='/etc/ca-certificates', help='The path to a CA_BUNDLE file or directory with certificates of trusted CAs.'), cfg.StrOpt('client_cert', sample_default='/etc/ca-certificates/service-client-keystore', help='Client side certificate, as a single file path containing either the certificate only or the private key and the certificate.'), cfg.StrOpt('client_key', help='Client side private key, in case client_cert is specified but does not includes the private key.'), cfg.StrOpt('timeout', default=60, help='Timeout is the number of seconds the request will wait for your client to establish a connection to a remote machine call on the socket.')]

    def list_options_for_discovery(self):
        return self._uri_driver_opts

    def open_source_from_opt_group(self, conf, group_name):
        conf.register_opts(self._uri_driver_opts, group_name)
        return URIConfigurationSource(conf[group_name].uri, conf[group_name].ca_path, conf[group_name].client_cert, conf[group_name].client_key, conf[group_name].timeout)