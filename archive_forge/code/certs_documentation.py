from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import files
import six
Read certificate_file and return the certificate in DER encoding.

  Args:
    certificate_file: A file handle to the certificate in PEM or DER format.

  Returns:
    The certificate in DER encoding.

  Raises:
    BadArgumentException: The provided certificate failed to parse as a PEM.
  