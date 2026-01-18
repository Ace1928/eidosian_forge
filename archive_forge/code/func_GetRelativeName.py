from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import iam_util
def GetRelativeName(resource, location):
    if location:
        resource_uri = resource.RelativeName()
        split = resource_uri.split('/')
        return '/'.join(split[:2]) + f'/locations/{location}/' + '/'.join(split[2:])
    return resource.RelativeName()