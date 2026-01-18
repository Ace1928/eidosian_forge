from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import uuid
from apitools.base.py import encoding as api_encoding
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.datastream import camel_case_utils
from googlecloudsdk.api_lib.datastream import exceptions as ds_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
import six
def UpdateV1alpha1ToV1MaskFields(field_mask):
    """Updates field mask paths according to the v1alpha1 > v1 Datastream API change.

  This allows for backwards compatibility with the current client field
  mask.

  Args:
    field_mask: List[str], list of stream fields to update

  Returns:
    updated_field_mask: List[str] field mask with fields translated
      from v1alpha1 API to v1.
  """
    updated_field_mask = []
    for path in field_mask:
        field_to_translate = None
        for field in _UPDATE_MASK_FIELD_TRANSLATION_V1ALPHA1_TO_V1:
            if field in path:
                field_to_translate = field
                break
        if field_to_translate:
            updated_field_mask.append(path.replace(field_to_translate, _UPDATE_MASK_FIELD_TRANSLATION_V1ALPHA1_TO_V1[field_to_translate]))
        else:
            updated_field_mask.append(path)
    return updated_field_mask