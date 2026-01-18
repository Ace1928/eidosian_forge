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
class MigrationResult(object):
    """The changes that are about to be applied on a declarative form.

  Args:
    new_files: {str, str} a mapping from absolute file path to new contents of
      the file, or None if the file should be deleted.
  """

    def __init__(self, new_files):
        self.new_files = new_files

    def __eq__(self, other):
        return self.new_files == other.new_files

    def __ne__(self, other):
        return not self == other

    def _Backup(self):
        for path in self.new_files.keys():
            bak_path = _Bakify(path)
            if not os.path.isfile(path):
                continue
            if os.path.exists(bak_path):
                raise MigrationError('Backup file path [{}] already exists.'.format(bak_path))
            log.err.Print('Copying [{}] to [{}]'.format(path, bak_path))
            shutil.copy2(path, bak_path)

    def _WriteFiles(self):
        for path, new_contents in self.new_files.items():
            if new_contents is None:
                log.err.Print('Deleting [{}]'.format(path))
                os.remove(path)
            else:
                log.err.Print('{} [{}]'.format('Overwriting' if os.path.exists(path) else 'Writing', path))
                files.WriteFileContents(path, new_contents)

    def Apply(self):
        """Backs up first, then deletes, overwrites and writes new files."""
        self._Backup()
        self._WriteFiles()