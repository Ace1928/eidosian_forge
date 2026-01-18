from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadCertificateFeatureState(_messages.Message):
    """WorkloadCertificateFeatureState describes the state of the workload
  certificate feature. This is required since FeatureStateDetails requires a
  state.
  """