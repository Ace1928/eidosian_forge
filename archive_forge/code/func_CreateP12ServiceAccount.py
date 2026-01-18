from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from google.auth import _helpers
from google.auth.crypt import base as crypt_base
from google.oauth2 import service_account
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import encoding
def CreateP12ServiceAccount(key_string, password=None, **kwargs):
    """Creates a service account from a p12 key and handles import errors."""
    log.warning('.p12 service account keys are not recommended unless it is necessary for backwards compatibility. Please switch to a newer .json service account key for this account.')
    try:
        return Credentials.from_service_account_pkcs12_keystring(key_string, password, **kwargs)
    except ImportError:
        if not encoding.GetEncodedValue(os.environ, 'CLOUDSDK_PYTHON_SITEPACKAGES'):
            raise MissingDependencyError('pyca/cryptography is not available. Please install or upgrade it to a version >= {} and set the environment variable CLOUDSDK_PYTHON_SITEPACKAGES to 1. If that does not work, see https://developers.google.com/cloud/sdk/crypto for details or consider using .json private key instead.'.format(_PYCA_CRYPTOGRAPHY_MIN_VERSION))
        else:
            raise MissingDependencyError('pyca/cryptography is not available or the version is < {}. Please install or upgrade it to a newer version. See https://developers.google.com/cloud/sdk/crypto for details or consider using .json private key instead.'.format(_PYCA_CRYPTOGRAPHY_MIN_VERSION))