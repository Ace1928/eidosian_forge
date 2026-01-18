from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class MigrateConfig(base.Group):
    """Convert configuration files from one format to another.

  Automated one-time migration tooling for helping with transition of
  configuration from one state to another. Currently exclusively
  provides commands for converting datastore-indexes.xml, queue.xml, cron.xml
  and dispatch.xml to their yaml counterparts.
  """
    category = base.APP_ENGINE_CATEGORY
    detailed_help = {'EXAMPLES': '          To convert a cron.xml to cron.yaml, run:\n\n            $ {command} cron-xml-to-yaml my/app/WEB-INF/cron.xml\n      '}