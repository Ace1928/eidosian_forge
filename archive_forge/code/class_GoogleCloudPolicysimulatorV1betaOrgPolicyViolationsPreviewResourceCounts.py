from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1betaOrgPolicyViolationsPreviewResourceCounts(_messages.Message):
    """A summary of the state of all resources scanned for compliance with the
  changed OrgPolicy.

  Fields:
    compliant: Output only. Number of scanned resources with zero violations.
    errors: Output only. Number of resources that returned an error when
      scanned.
    noncompliant: Output only. Number of scanned resources with at least one
      violation.
    scanned: Output only. Number of resources checked for compliance. Must
      equal: unenforced + noncompliant + compliant + error
    unenforced: Output only. Number of resources where the constraint was not
      enforced, i.e. the Policy set `enforced: false` for that resource.
  """
    compliant = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    errors = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    noncompliant = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    scanned = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    unenforced = _messages.IntegerField(5, variant=_messages.Variant.INT32)