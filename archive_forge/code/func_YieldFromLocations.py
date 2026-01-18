from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as api_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.functions.v1 import util
from googlecloudsdk.calliope import exceptions as base_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def YieldFromLocations(locations, project, limit, messages, client):
    """Yield the functions from the given locations."""
    for location in locations:
        location_ref = resources.REGISTRY.Parse(location, params={'projectsId': project}, collection='cloudfunctions.projects.locations')
        for function in _YieldFromLocation(location_ref, limit, messages, client):
            yield function