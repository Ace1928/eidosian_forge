from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.security_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.security_policies import flags
from googlecloudsdk.command_lib.compute.security_policies import security_policies_utils
Add a user defined field to a Compute Engine security policy.

  *{command}* is used to add user defined fields to security policies.

  ## EXAMPLES

  To add a user defined field run this:

    $ {command} SECURITY_POLICY \
       --user-defined-field-name=my-field \
       --base=ipv6 \
       --offset=10 \
       --size=3
  