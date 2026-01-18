from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.util import exceptions as api_lib_util_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.projects import util as command_lib_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import store as c_store
def WarnIfSettingNonExistentRegionZone(value, zonal=True):
    """Warn if setting 'compute/region' or 'compute/zone' to wrong value."""
    zonal_msg = '{} is not a valid zone. Run `gcloud compute zones list` to get all zones.'.format(value)
    regional_msg = '{} is not a valid region. Run `gcloud compute regions list`to get all regions.'.format(value)
    if not value:
        log.warning(zonal_msg if zonal else regional_msg)
        return True
    holder = base_classes.ComputeApiHolder(base.ReleaseTrack.GA)
    client = holder.client
    zone_request = [(client.apitools_client.zones, 'Get', client.messages.ComputeZonesGetRequest(project=properties.VALUES.core.project.GetOrFail(), zone=value))]
    region_request = [(client.apitools_client.regions, 'Get', client.messages.ComputeRegionsGetRequest(project=properties.VALUES.core.project.GetOrFail(), region=value))]
    try:
        errors = []
        client.MakeRequests(zone_request if zonal else region_request, errors)
        if errors and 404 in errors[0]:
            log.warning(zonal_msg if zonal else regional_msg)
            return True
    except (calliope_exceptions.ToolException, apitools_exceptions.HttpError, c_store.NoCredentialsForAccountException, api_lib_util_exceptions.HttpException):
        pass
    log.warning('Property validation for compute/{} was skipped.'.format('zone' if zonal else 'region'))
    return False