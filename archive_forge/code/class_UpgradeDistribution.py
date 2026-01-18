from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpgradeDistribution(_messages.Message):
    """The Upgrade Distribution represents metadata about the Upgrade for each
  operating system (CPE). Some distributions have additional metadata around
  updates, classifying them into various categories and severities.

  Fields:
    classification: The operating system classification of this Upgrade, as
      specified by the upstream operating system upgrade feed. For Windows the
      classification is one of the category_ids listed at
      https://docs.microsoft.com/en-us/previous-
      versions/windows/desktop/ff357803(v=vs.85)
    cpeUri: Required - The specific operating system this metadata applies to.
      See https://cpe.mitre.org/specification/.
    cve: The cve tied to this Upgrade.
    severity: The severity as specified by the upstream operating system.
  """
    classification = _messages.StringField(1)
    cpeUri = _messages.StringField(2)
    cve = _messages.StringField(3, repeated=True)
    severity = _messages.StringField(4)