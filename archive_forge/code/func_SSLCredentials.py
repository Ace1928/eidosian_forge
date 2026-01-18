from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import atexit
import enum
import json
import os
from google.auth import exceptions as google_auth_exceptions
from google.auth.transport import _mtls_helper
from googlecloudsdk.command_lib.auth import enterprise_certificate_config
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def SSLCredentials(config_path):
    """Generates the client SSL credentials.

  Args:
    config_path: path to the context aware configuration file.

  Raises:
    CertProvisionException: if the cert could not be provisioned.
    ConfigException: if there is an issue in the context aware config.

  Returns:
    Tuple[bytes, bytes]: client certificate and private key bytes in PEM format.
  """
    try:
        has_cert, cert_bytes, key_bytes, _ = _mtls_helper.get_client_ssl_credentials(generate_encrypted_key=False, context_aware_metadata_path=config_path)
        if has_cert:
            return (cert_bytes, key_bytes)
    except google_auth_exceptions.ClientCertError as caught_exc:
        new_exc = CertProvisionException(caught_exc)
        six.raise_from(new_exc, caught_exc)
    raise ConfigException()