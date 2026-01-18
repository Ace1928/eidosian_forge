from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.app.api import appengine_api_client_base as base
from googlecloudsdk.calliope import base as calliope_base
def GetApiClientForTrack(release_track):
    api_version = VERSION_MAP[release_track]
    return AppengineFirewallApiClient.GetApiClient(api_version)