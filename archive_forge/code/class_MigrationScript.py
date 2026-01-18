from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shutil
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.third_party.appengine.datastore import datastore_index_xml
from googlecloudsdk.third_party.appengine.tools import cron_xml_parser
from googlecloudsdk.third_party.appengine.tools import dispatch_xml_parser
from googlecloudsdk.third_party.appengine.tools import queue_xml_parser
class MigrationScript(object):
    """Object representing a migration script and its metadata.

  Attributes:
    migrate_fn: a function which accepts a variable number of self-defined
      kwargs and returns a MigrationResult.
    description: str, description for help texts and prompts.
  """

    def __init__(self, migrate_fn, description):
        self.migrate_fn = migrate_fn
        self.description = description