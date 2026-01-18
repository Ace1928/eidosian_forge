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
class _EnterpriseCertConfigImpl(_ConfigImpl):
    """Represents the configurations associated with context aware access through a enterprise certificate on TPM or OS key store."""

    def __init__(self, certificate_config_file_path):
        super(_EnterpriseCertConfigImpl, self).__init__(ConfigType.ENTERPRISE_CERTIFICATE)
        self.certificate_config_file_path = certificate_config_file_path