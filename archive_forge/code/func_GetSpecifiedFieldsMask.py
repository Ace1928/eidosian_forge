from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import http_encoding
import six  # pylint: disable=unused-import
from six.moves import filter  # pylint:disable=redefined-builtin
from six.moves import map  # pylint:disable=redefined-builtin
def GetSpecifiedFieldsMask(args, queue_type, release_track=base.ReleaseTrack.GA):
    """Returns the mask fields to use with the given args."""
    updatable_config = QueueUpdatableConfiguration.FromQueueTypeAndReleaseTrack(queue_type, release_track)
    specified_args = _SpecifiedArgs(args, updatable_config.AllConfigs(), clear_args=True)
    args_to_mask = updatable_config.GetConfigToUpdateMaskMapping()
    masks_field = [args_to_mask[arg] for arg in specified_args]
    if hasattr(args, 'type') and args.type == constants.PULL_TASK:
        masks_field.append('type')
    return sorted(set(masks_field))