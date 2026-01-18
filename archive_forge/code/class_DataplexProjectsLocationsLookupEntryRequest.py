from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLookupEntryRequest(_messages.Message):
    """A DataplexProjectsLocationsLookupEntryRequest object.

  Enums:
    ViewValueValuesEnum: Optional. View for controlling which parts of an
      entry are to be returned.

  Fields:
    aspectTypes: Optional. Limits the aspects returned to the provided aspect
      types. Only works if the CUSTOM view is selected.
    entry: Required. The resource name of the Entry: projects/{project}/locati
      ons/{location}/entryGroups/{entry_group}/entries/{entry}.
    name: Required. The project to which the request should be attributed in
      the following form: projects/{project}/locations/{location}.
    paths: Optional. Limits the aspects returned to those associated with the
      provided paths within the Entry. Only works if the CUSTOM view is
      selected.
    view: Optional. View for controlling which parts of an entry are to be
      returned.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional. View for controlling which parts of an entry are to be
    returned.

    Values:
      ENTRY_VIEW_UNSPECIFIED: Unspecified EntryView. Defaults to FULL.
      BASIC: Returns entry only, without aspects.
      FULL: Returns all required aspects as well as the keys of all non-
        required aspects.
      CUSTOM: Returns aspects matching custom fields in GetEntryRequest. If
        the number of aspects would exceed 100, the first 100 will be
        returned.
      ALL: Returns all aspects. If the number of aspects would exceed 100, the
        first 100 will be returned.
    """
        ENTRY_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
        CUSTOM = 3
        ALL = 4
    aspectTypes = _messages.StringField(1, repeated=True)
    entry = _messages.StringField(2)
    name = _messages.StringField(3, required=True)
    paths = _messages.StringField(4, repeated=True)
    view = _messages.EnumField('ViewValueValuesEnum', 5)