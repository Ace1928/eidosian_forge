from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import exceptions
from googlecloudsdk.api_lib.resource_manager import org_policies
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import org_policies_base
from googlecloudsdk.command_lib.resource_manager import org_policies_flags as flags
import six
Add values to an Organization Policy denied_values list policy.

  Adds one or more values to the specified Organization Policy denied_values
  list policy associated with the specified resource.

  ## EXAMPLES

  The following command adds `devEnv` and `prodEnv` to an Organization Policy
  denied_values list policy for constraint `serviceuser.services`
  on project `foo-project`:

    $ {command} serviceuser.services --project=foo-project devEnv prodEnv
  