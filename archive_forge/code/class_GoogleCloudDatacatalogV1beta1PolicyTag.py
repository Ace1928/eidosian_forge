from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1PolicyTag(_messages.Message):
    """Denotes one policy tag in a taxonomy (e.g. ssn). Policy Tags can be
  defined in a hierarchy. For example, consider the following hierarchy:
  Geolocation -> (LatLong, City, ZipCode). PolicyTag "Geolocation" contains
  three child policy tags: "LatLong", "City", and "ZipCode".

  Fields:
    childPolicyTags: Output only. Resource names of child policy tags of this
      policy tag.
    description: Description of this policy tag. It must: contain only unicode
      characters, tabs, newlines, carriage returns and page breaks; and be at
      most 2000 bytes long when encoded in UTF-8. If not set, defaults to an
      empty description. If not set, defaults to an empty description.
    displayName: Required. User defined name of this policy tag. It must: be
      unique within the parent taxonomy; contain only unicode letters,
      numbers, underscores, dashes and spaces; not start or end with spaces;
      and be at most 200 bytes long when encoded in UTF-8.
    name: Identifier. Resource name of this policy tag, whose format is: "proj
      ects/{project_number}/locations/{location_id}/taxonomies/{taxonomy_id}/p
      olicyTags/{id}".
    parentPolicyTag: Resource name of this policy tag's parent policy tag
      (e.g. for the "LatLong" policy tag in the example above, this field
      contains the resource name of the "Geolocation" policy tag). If empty,
      it means this policy tag is a top level policy tag (e.g. this field is
      empty for the "Geolocation" policy tag in the example above). If not
      set, defaults to an empty string.
  """
    childPolicyTags = _messages.StringField(1, repeated=True)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    name = _messages.StringField(4)
    parentPolicyTag = _messages.StringField(5)