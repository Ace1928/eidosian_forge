from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import dataclasses
import functools
import random
import string
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.run import condition as run_condition
from googlecloudsdk.api_lib.run import configuration
from googlecloudsdk.api_lib.run import domain_mapping
from googlecloudsdk.api_lib.run import execution
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run import job
from googlecloudsdk.api_lib.run import metric_names
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import route
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import task
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.run import artifact_registry
from googlecloudsdk.command_lib.run import config_changes as config_changes_mod
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import messages_util
from googlecloudsdk.command_lib.run import name_generator
from googlecloudsdk.command_lib.run import op_pollers
from googlecloudsdk.command_lib.run import resource_name_conversion
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.run.sourcedeploys import deployer
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
import six
def AddOrRemoveIamPolicyBinding(self, service_ref, add_binding=True, member=None, role=None):
    """Add or remove the given IAM policy binding to the provided service.

    If no members or role are provided, set the IAM policy to the current IAM
    policy. This is useful for checking whether the authenticated user has
    the appropriate permissions for setting policies.

    Args:
      service_ref: str, The service to which to add the IAM policy.
      add_binding: bool, Whether to add to or remove from the IAM policy.
      member: str, One of the users for which the binding applies.
      role: str, The role to grant the provided members.

    Returns:
      A google.iam.v1.TestIamPermissionsResponse.
    """
    messages = self.messages_module
    oneplatform_service = resource_name_conversion.K8sToOnePlatform(service_ref, self._region)
    policy = self._GetIamPolicy(oneplatform_service)
    if member and role:
        if add_binding:
            iam_util.AddBindingToIamPolicy(messages.Binding, policy, member, role)
        elif iam_util.BindingInPolicy(policy, member, role):
            iam_util.RemoveBindingFromIamPolicy(policy, member, role)
    request = messages.RunProjectsLocationsServicesSetIamPolicyRequest(resource=six.text_type(oneplatform_service), setIamPolicyRequest=messages.SetIamPolicyRequest(policy=policy))
    result = self._op_client.projects_locations_services.SetIamPolicy(request)
    return result