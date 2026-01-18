from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1alpha3Category(_messages.Message):
    """Denotes one category in a taxonomy (e.g. ssn). Categories can be defined
  in a hierarchy. For example, consider the following hierachy: Geolocation |
  ------------------------------------ | | | LatLong City ZipCode Category
  "Geolocation" contains three child categories: "LatLong", "City", and
  "ZipCode".

  Fields:
    childCategoryIds: Output only. Ids of child categories of this category.
    description: Description of the category. The length of the description is
      limited to 2000 bytes when encoded in UTF-8.
    displayName: Required. Human readable name of this category. Max 200 bytes
      when encoded in UTF-8.
    name: Output only. Resource name of the category, whose format is:
      "projects/project_number/taxonomies/{taxonomy_id}/categories/{id}".
    parentCategoryId: Id of the parent category to this category (e.g. for
      category "LatLong" in the example above, this field contains the id of
      category "Geolocation"). If empty, it means this category is a top level
      category (e.g. this field is empty for category "Geolocation" in the
      example above).
  """
    childCategoryIds = _messages.StringField(1, repeated=True)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    name = _messages.StringField(4)
    parentCategoryId = _messages.StringField(5)