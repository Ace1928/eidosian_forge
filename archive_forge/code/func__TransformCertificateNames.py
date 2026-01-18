from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.certificate_manager import certificate_map_entries
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.certificate_manager import resource_args
from googlecloudsdk.command_lib.certificate_manager import util
def _TransformCertificateNames(cert_names, undefined=''):
    """Transforms fully qualified cert names to their IDs."""
    if not cert_names:
        return undefined
    result = []
    for name in cert_names:
        match = re.match(_CERT_NAME_REGEX, name)
        result.append(match.group(3) if match else name)
    return '\n'.join(result) if result else undefined