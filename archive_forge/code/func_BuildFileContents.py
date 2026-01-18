from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import property_selector
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.backend_services import backend_services_utils
from googlecloudsdk.command_lib.compute.backend_services import flags
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import edit
import six
def BuildFileContents(self, args, client, original_record, modifiable_record):
    buf = io.StringIO()
    for line in base_classes.HELP.splitlines():
        buf.write('#')
        if line:
            buf.write(' ')
        buf.write(line)
        buf.write('\n')
    buf.write('\n')
    buf.write(base_classes.SerializeDict(modifiable_record, args.format or Edit.DEFAULT_FORMAT))
    buf.write('\n')
    example = base_classes.SerializeDict(encoding.MessageToDict(self.GetExampleResource(client)), args.format or Edit.DEFAULT_FORMAT)
    base_classes.WriteResourceInCommentBlock(example, 'Example resource:', buf)
    buf.write('#\n')
    original = base_classes.SerializeDict(original_record, args.format or Edit.DEFAULT_FORMAT)
    base_classes.WriteResourceInCommentBlock(original, 'Original resource:', buf)
    return buf.getvalue()