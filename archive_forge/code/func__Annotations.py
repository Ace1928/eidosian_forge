from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkemulticloud import util
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _Annotations(self, args, parent_type):
    """Parses the annotations from the args.

    Args:
      args: Arguments to be parsed.
      parent_type: Type of the parent object.

    Returns:
      Returns the parsed annotations.
    """
    annotations = flags.GetAnnotations(args)
    if not annotations:
        return None
    annotation_type = parent_type.AnnotationsValue.AdditionalProperty
    return parent_type.AnnotationsValue(additionalProperties=[annotation_type(key=k, value=v) for k, v in annotations.items()])