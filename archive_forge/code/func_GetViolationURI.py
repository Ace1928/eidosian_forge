from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.assured import message_util
from googlecloudsdk.api_lib.assured import util
from googlecloudsdk.core import resources
def GetViolationURI(resource):
    violation = resources.REGISTRY.ParseRelativeName(resource.name, collection='assuredworkloads.organizations.locations.workloads.violations')
    return violation.SelfLink()