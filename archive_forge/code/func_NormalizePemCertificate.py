from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.certificate_manager import api_client
from googlecloudsdk.core.util import times
def NormalizePemCertificate(pem_certificate):
    """Normalizes the PEM certificate for the comparison by removing all whitespace characters.

  Args:
    pem_certificate: PEM certificate to be normalized.

  Returns:
    PEM certificate without whitespace characters.
  """
    return re.sub('\\s+', '', pem_certificate, flags=re.ASCII)