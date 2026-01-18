from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DeltaPresenceEstimationConfig(_messages.Message):
    """\\u03b4-presence metric, used to estimate how likely it is for an
  attacker to figure out that one given individual appears in a de-identified
  dataset. Similarly to the k-map metric, we cannot compute \\u03b4-presence
  exactly without knowing the attack dataset, so we use a statistical model
  instead.

  Fields:
    auxiliaryTables: Several auxiliary tables can be used in the analysis.
      Each custom_tag used to tag a quasi-identifiers field must appear in
      exactly one field of one auxiliary table.
    quasiIds: Required. Fields considered to be quasi-identifiers. No two
      fields can have the same tag.
    regionCode: ISO 3166-1 alpha-2 region code to use in the statistical
      modeling. Set if no column is tagged with a region-specific InfoType
      (like US_ZIP_5) or a region code.
  """
    auxiliaryTables = _messages.MessageField('GooglePrivacyDlpV2StatisticalTable', 1, repeated=True)
    quasiIds = _messages.MessageField('GooglePrivacyDlpV2QuasiId', 2, repeated=True)
    regionCode = _messages.StringField(3)