from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import models
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.ml_engine import region_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def AddIamPolicyBindingWithCondition(models_client, model, member, role, condition):
    """Adds IAM binding with condition to ml engine model's IAM policy."""
    model_ref = ParseModel(model)
    policy = models_client.GetIamPolicy(model_ref)
    iam_util.AddBindingToIamPolicyWithCondition(models_client.messages.GoogleIamV1Binding, models_client.messages.GoogleTypeExpr, policy, member, role, condition)
    return models_client.SetIamPolicy(model_ref, policy, 'bindings,etag')