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
def _MigrateDatastoreIndexXML(src, dst, auto_src=None):
    """Migration script for datastore-indexes.xml."""
    xml_str = files.ReadFileContents(src)
    indexes = datastore_index_xml.IndexesXmlToIndexDefinitions(xml_str)
    new_files = {src: None}
    if auto_src:
        xml_str_2 = files.ReadFileContents(auto_src)
        auto_indexes = datastore_index_xml.IndexesXmlToIndexDefinitions(xml_str_2)
        indexes.indexes += auto_indexes.indexes
        new_files[auto_src] = None
    new_files[dst] = indexes.ToYAML()
    return MigrationResult(new_files)