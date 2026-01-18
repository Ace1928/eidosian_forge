from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def AnalyzePackagesBeta(project, location, resource_uri, packages):
    """Make an RPC to the On-Demand Scanning v1beta1 AnalyzePackages method."""
    client = GetClient('v1beta1')
    messages = GetMessages('v1beta1')
    req = messages.OndemandscanningProjectsLocationsScansAnalyzePackagesRequest(parent=PARENT_TEMPLATE.format(project, location), analyzePackagesRequest=messages.AnalyzePackagesRequest(packages=packages, resourceUri=resource_uri))
    return client.projects_locations_scans.AnalyzePackages(req)