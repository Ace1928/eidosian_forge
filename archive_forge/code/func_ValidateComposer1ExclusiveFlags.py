from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import flags
from googlecloudsdk.command_lib.composer import image_versions_util
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
import six
def ValidateComposer1ExclusiveFlags(self, args, is_composer_v1, release_track):
    """Raises InputError if flags from Composer v2 are used when creating v1."""
    if args.python_version and (not is_composer_v1):
        raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V2_ERROR_MSG.format(opt='python-version'))
    if args.disk_size and (not is_composer_v1):
        raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V2_ERROR_MSG.format(opt='disk-size'))
    if args.machine_type and (not is_composer_v1):
        raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V2_ERROR_MSG.format(opt='machine-type'))