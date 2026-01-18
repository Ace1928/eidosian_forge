from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.concepts import base
from googlecloudsdk.command_lib.concepts import dependency_managers
from googlecloudsdk.command_lib.concepts import names
import six
A lazy property to use during concept parsing.

    Returns:
      googlecloudsdk.calliope.parser_extensions.Namespace: the parsed argparse
        namespace | None, if the parser hasn't been registered to the namespace
        yet.
    