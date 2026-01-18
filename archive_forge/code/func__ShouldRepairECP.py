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
def _ShouldRepairECP(cert_config):
    """Check if ECP binaries should be installed and the ECP config updated to point to them."""
    args = argv_utils.GetDecodedArgv()
    if 'init' in args:
        return False
    if 'cert_configs' not in cert_config:
        return False
    if len(cert_config['cert_configs'].keys()) < 1:
        return False
    if 'libs' not in cert_config:
        return False
    expected_keys = set(['ecp', 'ecp_client', 'tls_offload'])
    actual_keys = set(cert_config['libs'].keys())
    if expected_keys == actual_keys:
        return False
    return True