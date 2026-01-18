from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.services import enable_api
import six
def _HasValidKey(build):
    """Check whether a build provenance contains valid signature and key id.

  Args:
    build: container analysis build occurrence.

  Returns:
    A boolean value indicating whether build occurrence contains valid signature
    and key id.
  """
    if build and hasattr(build, 'envelope') and hasattr(build.envelope, 'signatures') and build.envelope.signatures:
        key_id_pattern = '^projects/verified-builder/locations/.+/keyRings/attestor/cryptoKeys/builtByGCB/cryptoKeyVersions/1$'

        def CheckSignature(signature):
            return hasattr(signature, 'sig') and signature.sig and hasattr(signature, 'keyid') and re.match(key_id_pattern, signature.keyid)
        filtered = filter(CheckSignature, build.envelope.signatures)
        if filtered:
            return True
    return False