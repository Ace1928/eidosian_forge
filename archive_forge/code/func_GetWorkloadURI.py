from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.assured import util
from googlecloudsdk.core import resources
def GetWorkloadURI(resource):
    workload = resources.REGISTRY.ParseRelativeName(resource.name, collection='assuredworkloads.organizations.locations.operations')
    return workload.SelfLink()