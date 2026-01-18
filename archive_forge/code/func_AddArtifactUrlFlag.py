from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.binauthz import arg_parsers
from googlecloudsdk.command_lib.kms import flags as kms_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs as presentation_specs_lib
def AddArtifactUrlFlag(parser, required=True):
    parser.add_argument('--artifact-url', required=required, type=str, help='Container URL. May be in the `gcr.io/repository/image` format, or may optionally contain the `http` or `https` scheme')