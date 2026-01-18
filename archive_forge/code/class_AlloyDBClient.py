from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
class AlloyDBClient(object):
    """Wrapper for alloydb API client and associated resources."""

    def __init__(self, release_track):
        api_version = VERSION_MAP[release_track]
        self.release_track = release_track
        self.alloydb_client = apis.GetClientInstance('alloydb', api_version)
        self.alloydb_messages = self.alloydb_client.MESSAGES_MODULE
        self.resource_parser = resources.Registry()
        self.resource_parser.RegisterApiByName('alloydb', api_version)