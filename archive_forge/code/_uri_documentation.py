import requests
import tempfile
from oslo_config import cfg
from oslo_config import sources
A configuration source for remote files served through http[s].

    :param uri: The Uniform Resource Identifier of the configuration to be
          retrieved.

    :param ca_path: The path to a CA_BUNDLE file or directory with
              certificates of trusted CAs.

    :param client_cert: Client side certificate, as a single file path
                  containing either the certificate only or the
                  private key and the certificate.

    :param client_key: Client side private key, in case client_cert is
                 specified but does not includes the private key.
    