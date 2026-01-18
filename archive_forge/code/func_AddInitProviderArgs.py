from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core.util import files
def AddInitProviderArgs(parser):
    """Add args for init provider."""
    zone = calliope_base.Argument('--zone', required=False, help='Default Google Cloud Zone for Zonal Resources.\n        If not specified the current `compute/zone` property will be used.')
    region = calliope_base.Argument('--region', required=False, help='Default Google Cloud Region for Regional Resources.\n      If not specified the current `compute/region` property will be used.')
    billing_group = parser.add_group(help='The below flags specify how the optional `user_project_override` and `billing_project` settings are configured for the Google Terraform Provider.\n      See the [Google Terraform Provider Config Reference](https://registry.terraform.io/providers/hashicorp/google/latest/docs/guides/provider_reference#user_project_override) for more details.', required=False, mutex=True)
    billing_group.add_argument('--use-gcloud-billing-project', action='store_true', help='If specified, will set `user_project_override` value in the Terrafom provider config to `true` and\n      set `billing_project` to the current gcloud `billing/quota_project` property.', default=False, required=False)
    billing_account_group = billing_group.add_group(help='Account Override Flags.')
    billing_account_group.add_argument('--tf-user-project-override', action='store_true', help='If specified, sets the `user_project_override` value in the Terraform provider config to `true`.', default=False, required=True)
    billing_account_group.add_argument('--tf-billing-project', help='If specified, sets the `billing_project` value in the Terraform provider config.', required=False)
    zone.AddToParser(parser)
    region.AddToParser(parser)