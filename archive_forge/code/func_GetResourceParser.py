from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def GetResourceParser(release_track=base.ReleaseTrack.GA):
    resource_parser = resources.Registry()
    api_version = VERSION_MAP.get(release_track)
    resource_parser.RegisterApiByName('edgenetwork', api_version)
    return resource_parser