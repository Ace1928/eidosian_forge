from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2KMapEstimationConfig(_messages.Message):
    """Reidentifiability metric. This corresponds to a risk model similar to
  what is called "journalist risk" in the literature, except the attack
  dataset is statistically modeled instead of being perfectly known. This can
  be done using publicly available data (like the US Census), or using a
  custom statistical model (indicated as one or several BigQuery tables), or
  by extrapolating from the distribution of values in the input dataset.

  Fields:
    auxiliaryTables: Several auxiliary tables can be used in the analysis.
      Each custom_tag used to tag a quasi-identifiers column must appear in
      exactly one column of one auxiliary table.
    quasiIds: Required. Fields considered to be quasi-identifiers. No two
      columns can have the same tag.
    regionCode: ISO 3166-1 alpha-2 region code to use in the statistical
      modeling. Set if no column is tagged with a region-specific InfoType
      (like US_ZIP_5) or a region code.
  """
    auxiliaryTables = _messages.MessageField('GooglePrivacyDlpV2AuxiliaryTable', 1, repeated=True)
    quasiIds = _messages.MessageField('GooglePrivacyDlpV2TaggedField', 2, repeated=True)
    regionCode = _messages.StringField(3)