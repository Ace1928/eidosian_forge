from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.privateca import operations
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.core import log
Enable a root certificate authority.

    Enables a root certificate authority. The root certificate authority will be
    allowed to issue certificates once enabled.

    ## EXAMPLES

    To enable a root CA:

        $ {command} prod-root --location=us-west1 --pool=my-pool
  