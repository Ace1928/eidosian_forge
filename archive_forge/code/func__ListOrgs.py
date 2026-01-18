from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib import apigee
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.apigee import argument_groups
from googlecloudsdk.command_lib.apigee import defaults
from googlecloudsdk.command_lib.apigee import prompts
from googlecloudsdk.command_lib.apigee import resource_args
from googlecloudsdk.core.console import console_io
def _ListOrgs():
    response = apigee.OrganizationsClient.List()
    if 'organizations' in response:
        return [item['organization'] for item in response['organizations']]
    else:
        return []