from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadcertificateProjectsLocationsWorkloadRegistrationsGetRequest(_messages.Message):
    """A WorkloadcertificateProjectsLocationsWorkloadRegistrationsGetRequest
  object.

  Fields:
    name: Required. Name of the resource. Format: `projects/{project ID or num
      ber}/locations/{location}/workloadRegistrations/{workload_registration_i
      d}`
  """
    name = _messages.StringField(1, required=True)