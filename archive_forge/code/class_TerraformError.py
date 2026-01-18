from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TerraformError(_messages.Message):
    """Errors encountered during actuation using Terraform

  Fields:
    error: Original error response from underlying Google API, if available.
    errorDescription: A human-readable error description.
    httpResponseCode: HTTP response code returned from Google Cloud Platform
      APIs when Terraform fails to provision the resource. If unset or 0, no
      HTTP response code was returned by Terraform.
    resourceAddress: Address of the resource associated with the error, e.g.
      `google_compute_network.vpc_network`.
  """
    error = _messages.MessageField('Status', 1)
    errorDescription = _messages.StringField(2)
    httpResponseCode = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    resourceAddress = _messages.StringField(4)