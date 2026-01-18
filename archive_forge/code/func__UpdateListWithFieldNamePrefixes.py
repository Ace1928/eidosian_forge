from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datastream import exceptions as ds_exceptions
from googlecloudsdk.api_lib.datastream import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
def _UpdateListWithFieldNamePrefixes(self, update_fields, prefix_to_check, prefix_to_add):
    """Returns an updated list of field masks with necessary prefixes."""
    temp_fields = [prefix_to_add + field for field in update_fields if field.startswith(prefix_to_check)]
    update_fields = [x for x in update_fields if not x.startswith(prefix_to_check)]
    update_fields.extend(temp_fields)
    return update_fields