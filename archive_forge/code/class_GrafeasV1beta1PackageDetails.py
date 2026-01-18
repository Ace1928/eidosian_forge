from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrafeasV1beta1PackageDetails(_messages.Message):
    """Details of a package occurrence.

  Fields:
    installation: Required. Where the package was installed.
  """
    installation = _messages.MessageField('Installation', 1)