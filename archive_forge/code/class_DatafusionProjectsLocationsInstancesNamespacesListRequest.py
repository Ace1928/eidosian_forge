from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatafusionProjectsLocationsInstancesNamespacesListRequest(_messages.Message):
    """A DatafusionProjectsLocationsInstancesNamespacesListRequest object.

  Enums:
    ViewValueValuesEnum: By default, only basic information about a namespace
      is returned (e.g. name). When `NAMESPACE_VIEW_FULL` is specified,
      additional information associated with a namespace gets returned (e.g.
      IAM policy set on the namespace)

  Fields:
    pageSize: The maximum number of items to return.
    pageToken: The next_page_token value to use if there are additional
      results to retrieve for this list request.
    parent: Required. The instance to list its namespaces.
    view: By default, only basic information about a namespace is returned
      (e.g. name). When `NAMESPACE_VIEW_FULL` is specified, additional
      information associated with a namespace gets returned (e.g. IAM policy
      set on the namespace)
  """

    class ViewValueValuesEnum(_messages.Enum):
        """By default, only basic information about a namespace is returned (e.g.
    name). When `NAMESPACE_VIEW_FULL` is specified, additional information
    associated with a namespace gets returned (e.g. IAM policy set on the
    namespace)

    Values:
      NAMESPACE_VIEW_UNSPECIFIED: Default/unset value, which will use BASIC
        view.
      NAMESPACE_VIEW_BASIC: Show the most basic metadata of a namespace
      NAMESPACE_VIEW_FULL: Returns all metadata of a namespace
    """
        NAMESPACE_VIEW_UNSPECIFIED = 0
        NAMESPACE_VIEW_BASIC = 1
        NAMESPACE_VIEW_FULL = 2
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 4)