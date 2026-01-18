from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.alloydb import api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def _matchesTarget(self, target, cluster_id):
    pattern = 'projects/[^/]*/locations/[^/]*/clusters/' + cluster_id + '($|/.*$)'
    return re.match(pattern, target) is not None