from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetEncryptionKmsKeyNameArg():
    return base.Argument('--encryption-kms-key-name', required=False, default=None, help='      The Cloud KMS resource identifier of the customer managed encryption key\n      used to protect a resource. Has the form:\n      projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key.\n\n      The key needs to be in the same region as where the compute resource is\n      created.\n      ')