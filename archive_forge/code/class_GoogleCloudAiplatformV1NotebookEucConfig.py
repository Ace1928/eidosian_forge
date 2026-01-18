from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1NotebookEucConfig(_messages.Message):
    """The euc configuration of NotebookRuntimeTemplate.

  Fields:
    bypassActasCheck: Output only. Whether ActAs check is bypassed for service
      account attached to the VM. If false, we need ActAs check for the
      default Compute Engine Service account. When a Runtime is created, a VM
      is allocated using Default Compute Engine Service Account. Any user
      requesting to use this Runtime requires Service Account User (ActAs)
      permission over this SA. If true, Runtime owner is using EUC and does
      not require the above permission as VM no longer use default Compute
      Engine SA, but a P4SA.
    eucDisabled: Input only. Whether EUC is disabled in this
      NotebookRuntimeTemplate. In proto3, the default value of a boolean is
      false. In this way, by default EUC will be enabled for
      NotebookRuntimeTemplate.
  """
    bypassActasCheck = _messages.BooleanField(1)
    eucDisabled = _messages.BooleanField(2)