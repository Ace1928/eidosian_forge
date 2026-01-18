from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudHealthcareV1alpha2FhirImportResourcesResponse(_messages.Message):
    """Final response of importing resources. This structure is included in the
  response to describe the detailed outcome. It is only included when the
  operation finishes successfully.
  """