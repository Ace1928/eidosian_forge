from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadCertificates(_messages.Message):
    """Configuration for issuance of mTLS keys and certificates to Kubernetes
  pods.

  Fields:
    enableCertificates: enable_certificates controls issuance of workload mTLS
      certificates. If set, the GKE Workload Identity Certificates controller
      and node agent will be deployed in the cluster, which can then be
      configured by creating a WorkloadCertificateConfig Custom Resource.
      Requires Workload Identity (workload_pool must be non-empty).
  """
    enableCertificates = _messages.BooleanField(1)