from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1alpha3Taxonomy(_messages.Message):
    """A taxonomy is a collection of categories of business significance,
  typically associated with the substance of the category (e.g. credit card,
  SSN), or how it is used (e.g. account name, user ID).

  Fields:
    description: Description of the taxonomy. The length of the description is
      limited to 2000 bytes when encoded in UTF-8.
    displayName: Required. Human readable name of this taxonomy. Max 200 bytes
      when encoded in UTF-8.
    name: Output only. Resource name of the taxonomy, whose format is:
      "projects/project_number/taxonomies/{id}".
  """
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    name = _messages.StringField(3)