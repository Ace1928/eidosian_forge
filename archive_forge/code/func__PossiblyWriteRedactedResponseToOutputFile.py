from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _PossiblyWriteRedactedResponseToOutputFile(value, parsed_args):
    """Helper function for writing redacted contents to an output file."""
    if not parsed_args.output_file:
        return
    with files.BinaryFileWriter(parsed_args.output_file) as outfile:
        outfile.write(value)
    log.status.Print('The redacted contents can be viewed in [{}]'.format(parsed_args.output_file))