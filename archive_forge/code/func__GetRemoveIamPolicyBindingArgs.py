from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import models
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.ml_engine import endpoint_util
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import models_util
from googlecloudsdk.command_lib.ml_engine import region_util
def _GetRemoveIamPolicyBindingArgs(parser, add_condition=False):
    iam_util.AddArgsForRemoveIamPolicyBinding(parser, add_condition=add_condition)
    flags.GetModelResourceArg(required=True, verb='for which to remove IAM policy binding from').AddToParser(parser)
    flags.GetRegionArg(include_global=True).AddToParser(parser)
    base.URI_FLAG.RemoveFromParser(parser)