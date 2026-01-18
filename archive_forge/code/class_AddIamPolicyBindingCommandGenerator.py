from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import sys
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py.exceptions import HttpBadRequestError
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.apis import yaml_command_schema
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import files
import six
class AddIamPolicyBindingCommandGenerator(BaseIamPolicyBindingCommandGenerator):
    """Generator for add-iam-policy binding commands."""
    command_type = yaml_command_schema.CommandType.ADD_IAM_POLICY_BINDING

    def _GetModifiedIamPolicyAddIamBinding(self, args, add_condition=False):
        """Get the IAM policy and add the specified binding to it.

    Args:
      args: an argparse namespace.
      add_condition: True if support condition.

    Returns:
      IAM policy.
    """
        from googlecloudsdk.command_lib.iam import iam_util
        method = self.arg_generator.GetPrimaryResource(self.methods, args).method
        binding_message_type = method.GetMessageByName('Binding')
        if add_condition:
            condition = iam_util.ValidateAndExtractConditionMutexRole(args)
            policy = self._GetIamPolicy(args)
            condition_message_type = method.GetMessageByName('Expr')
            iam_util.AddBindingToIamPolicyWithCondition(binding_message_type, condition_message_type, policy, args.member, args.role, condition)
        else:
            policy = self._GetIamPolicy(args)
            iam_util.AddBindingToIamPolicy(binding_message_type, policy, args.member, args.role)
        return policy

    def _Generate(self):
        """Generates an add-iam-policy-binding command.

    An add-iam-policy-binding command adds a binding to a IAM policy. A
    binding consists of a member, a role to define the role of the member, and
    an optional condition to define in what condition the binding is valid.
    Two API methods are called to get and set the policy on the resource.

    Returns:
      calliope.base.Command, The command that implements the spec.
    """
        from googlecloudsdk.command_lib.iam import iam_util

        class Command(base.Command):
            """Add IAM policy binding command closure."""

            @staticmethod
            def Args(parser):
                iam_util.AddArgsForAddIamPolicyBinding(parser, role_completer=self._GenerateDeclarativeIamRolesCompleter(), add_condition=self._add_condition, hide_special_member_types=self._hide_special_member_types)
                self._CommonArgs(parser)
                base.URI_FLAG.RemoveFromParser(parser)

            def Run(self_, args):
                """Called when command is executed."""
                policy_request_path = 'setIamPolicyRequest'
                if self.spec.iam:
                    policy_request_path = self.spec.iam.set_iam_policy_request_path or policy_request_path
                policy_field_path = policy_request_path + '.policy'
                policy = self._GetModifiedIamPolicyAddIamBinding(args, add_condition=self._add_condition)
                if self.spec.iam and self.spec.iam.policy_version:
                    policy.version = self.spec.iam.policy_version
                self.spec.request.static_fields[policy_field_path] = policy
                try:
                    ref, response = self._CommonRun(args)
                except HttpBadRequestError as ex:
                    log.err.Print('ERROR: Policy modification failed. For a binding with condition, run "gcloud alpha iam policies lint-condition" to identify issues in condition.')
                    raise ex
                iam_util.LogSetIamPolicy(ref.Name(), self._GetDisplayResourceType(args))
                return self._HandleResponse(response, args)
        return Command