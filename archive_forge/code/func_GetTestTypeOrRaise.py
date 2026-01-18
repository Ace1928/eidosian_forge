from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import arg_file
from googlecloudsdk.api_lib.firebase.test import arg_util
from googlecloudsdk.api_lib.firebase.test import arg_validate
from googlecloudsdk.api_lib.firebase.test.ios import catalog_manager
from googlecloudsdk.calliope import exceptions
def GetTestTypeOrRaise(self, args):
    """If the test type is not user-specified, infer the most reasonable value.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        gcloud command invocation (i.e. group and command arguments combined).

    Returns:
      The type of the test to be run (e.g. 'xctest'), and also sets the 'type'
      arg if it was not user-specified.

    Raises:
      InvalidArgumentException if an explicit test type is invalid.
    """
    if not args.type:
        args.type = 'xctest'
    if args.type not in self._typed_arg_rules:
        raise exceptions.InvalidArgumentException('type', "'{0}' is not a valid test type.".format(args.type))
    return args.type