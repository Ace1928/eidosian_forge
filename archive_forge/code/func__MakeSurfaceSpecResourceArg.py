from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os.path
from googlecloudsdk.core import branding
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import name_parsing
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
def _MakeSurfaceSpecResourceArg(collection_info):
    """Makes resource arg name for surface specification context."""
    return name_parsing.convert_collection_name_to_delimited(collection_info.name, delimiter='_').upper()