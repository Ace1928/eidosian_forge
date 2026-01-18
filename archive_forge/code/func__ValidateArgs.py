from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import cryptokeyversions
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.privateca import certificate_utils
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.privateca import flags
from googlecloudsdk.command_lib.privateca import key_generation
from googlecloudsdk.command_lib.privateca import pem_utils
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
def _ValidateArgs(self, args):
    """Validates the command-line args."""
    if args.IsSpecified('use_preset_profile') and args.IsSpecified('template'):
        raise exceptions.OneOfArgumentsRequiredException(['--use-preset-profile', '--template'], 'To create a certificate, please specify either a preset profile or a certificate template.')
    resource_args.ValidateResourceIsCompleteIfSpecified(args, 'kms_key_version')