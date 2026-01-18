from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddServicesFlag():
    return base.Argument('--services', metavar='NAME=TAG', type=arg_parsers.ArgDict(), hidden=True, help='\n        The flag to be used with the --from-run-container flag to specify the\n        name of the service present in a given target.\n        This will be a repeated flag.\n\n        *target_id*::: The target_id.\n        *service*::: The name of the service in the specified target_id.\n\n        For example:\n\n          $gcloud deploy releases create foo \\\n              --from-run-container=path/to/image1:v1@sha256:45db24\n              --services=dev_target:dev_service\n              --services=prod_target:prod_service\n      ')