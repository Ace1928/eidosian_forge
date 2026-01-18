from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from typing import Iterator
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet import resources as fleet_resources
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as fleet_messages
def _AddBinauthzPolicyBindings(self, binary_authorization_config_group: parser_arguments.ArgumentInterceptor):
    platform_policy_type = arg_parsers.RegexpValidator(_BINAUTHZ_GKE_POLICY_REGEX, 'GKE policy resource names have the following format: `projects/{project_number}/platforms/gke/policies/{policy_id}`')
    binary_authorization_config_group.add_argument('--binauthz-policy-bindings', default=None, action='append', metavar='name=BINAUTHZ_POLICY', help=textwrap.dedent('          The relative resource name of the Binary Authorization policy to audit\n          and/or enforce. GKE policies have the following format:\n          `projects/{project_number}/platforms/gke/policies/{policy_id}`.'), type=arg_parsers.ArgDict(spec={'name': platform_policy_type}, required_keys=['name'], max_length=1))