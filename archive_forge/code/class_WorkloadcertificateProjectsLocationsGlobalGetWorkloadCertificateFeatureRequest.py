from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadcertificateProjectsLocationsGlobalGetWorkloadCertificateFeatureRequest(_messages.Message):
    """A WorkloadcertificateProjectsLocationsGlobalGetWorkloadCertificateFeatur
  eRequest object.

  Fields:
    name: Required. Name of the `WorkloadCertificateFeature` resource. Format:
      `projects/{project ID or
      number}/locations/global/workloadCertificateFeature`.
  """
    name = _messages.StringField(1, required=True)