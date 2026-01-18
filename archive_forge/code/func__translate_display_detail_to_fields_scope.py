from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import shim_format_util
import six
def _translate_display_detail_to_fields_scope(display_detail, is_bucket_listing):
    """Translates display details to fields scope equivalent.

  Args:
    display_detail (DisplayDetail): Argument to translate.
    is_bucket_listing (bool): Buckets require special handling.

  Returns:
    cloud_api.FieldsScope appropriate for the resources and display detail.
  """
    if display_detail == DisplayDetail.LONG and is_bucket_listing:
        return cloud_api.FieldsScope.SHORT
    display_detail_to_fields_scope = {DisplayDetail.SHORT: cloud_api.FieldsScope.SHORT, DisplayDetail.LONG: cloud_api.FieldsScope.NO_ACL, DisplayDetail.FULL: cloud_api.FieldsScope.FULL, DisplayDetail.JSON: cloud_api.FieldsScope.FULL}
    return display_detail_to_fields_scope[display_detail]