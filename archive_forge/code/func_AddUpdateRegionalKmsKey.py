from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.command_lib.secrets import completers as secrets_completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import resources
def AddUpdateRegionalKmsKey(parser: parser_arguments.ArgumentInterceptor, positional: bool=False, **kwargs) -> None:
    """Add flags for specifying regional cmek on secret updates.

  Args:
      parser: Given argument parser.
      positional : Whether the argument is positional.
      **kwargs: Extra arguments.
  """
    group = parser.add_group(mutex=True, help='regional kms key.', hidden=True)
    group.add_argument(_ArgOrFlag('regional-kms-key-name', positional), metavar='REGIONAL-KMS-KEY-NAME', help='regional kms key name for regional secret.', **kwargs)
    group.add_argument(_ArgOrFlag('remove-regional-kms-key-name', False), action='store_true', help='If set, removes the regional kms key.', **kwargs)