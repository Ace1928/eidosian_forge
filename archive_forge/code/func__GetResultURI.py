from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.cloudbuild import run_flags
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _GetResultURI(resource):
    result = resources.REGISTRY.ParseRelativeName(resource.name, collection='cloudbuild.projects.locations.results', api_version=client_util.GA_API_VERSION)
    return result.SelfLink()