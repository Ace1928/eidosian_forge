from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.privateca import flags
from googlecloudsdk.command_lib.privateca import operations
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
Disable a root certificate authority.

    Disables a root certificate authority. The root certificate authority
    will not be allowed to issue certificates once disabled. It may still revoke
    certificates and/or generate CRLs. The CA certfificate will still be
    included in the FetchCaCertificates response for the parent CA Pool.

    ## EXAMPLES

    To disable a root CA:

        $ {command} prod-root --pool=prod-root-pool --location=us-west1
  