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
def _WriteFiles(self):
    for path, new_contents in self.new_files.items():
        if new_contents is None:
            log.err.Print('Deleting [{}]'.format(path))
            os.remove(path)
        else:
            log.err.Print('{} [{}]'.format('Overwriting' if os.path.exists(path) else 'Writing', path))
            files.WriteFileContents(path, new_contents)