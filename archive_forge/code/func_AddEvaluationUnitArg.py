from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.binauthz import arg_parsers
from googlecloudsdk.command_lib.kms import flags as kms_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs as presentation_specs_lib
def AddEvaluationUnitArg(parser):
    """Adds a resource argument from file or from one or more images."""
    evaluation_unit_group = parser.add_group(mutex=True, required=True)
    evaluation_unit_group.add_argument('--resource', required=False, type=arg_parsers.ResourceFileName, help='The JSON or YAML file containing the Kubernetes resource to evaluate.')
    evaluation_unit_group.add_argument('--image', required=False, action='append', help='The image to evaluate. If the policy being evaluated has scoped checksets, this mode of evaluation will always use the default (unscoped) checkset.')