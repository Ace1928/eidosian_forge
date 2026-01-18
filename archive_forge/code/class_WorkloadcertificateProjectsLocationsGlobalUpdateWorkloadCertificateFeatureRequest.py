from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadcertificateProjectsLocationsGlobalUpdateWorkloadCertificateFeatureRequest(_messages.Message):
    """A WorkloadcertificateProjectsLocationsGlobalUpdateWorkloadCertificateFea
  tureRequest object.

  Fields:
    force: Force WorkloadCertificateFeature disablement. All
      WorkloadRegistrations in the same fleet will be deleted.
    name: Required. Name of the `WorkloadCertificateFeature` resource to
      update. Format: `projects/{project ID or
      number}/locations/global/workloadCertificateFeature`.
    workloadCertificateFeature: A WorkloadCertificateFeature resource to be
      passed as the request body.
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    workloadCertificateFeature = _messages.MessageField('WorkloadCertificateFeature', 3)