from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import files
class ListFromJson(base.ListCommand):
    """Read JSON data and list it on the standard output.

  *{command}* is a test harness for resource output formatting and filtering.
  It behaves like any other `gcloud ... list` command except that the resources
  are read from a JSON data file.

  The input JSON data is either a single resource object or a list of resource
  objects of the same type. The resources are printed on the standard output.
  The default output format is *json*.
  """

    @staticmethod
    def Args(parser):
        base.URI_FLAG.RemoveFromParser(parser)
        parser.add_argument('json_file', metavar='JSON-FILE', nargs='?', default=None, help='A file containing JSON data for a single resource or a list of resources of the same type. If omitted then the standard input is read.')
        parser.display_info.AddFormat('json')
        parser.display_info.AddCacheUpdater(None)

    def Run(self, args):
        if args.json_file:
            try:
                resources = json.loads(files.ReadFileContents(args.json_file))
            except (files.Error, ValueError) as e:
                raise exceptions.BadFileException('Cannot read [{}]: {}'.format(args.json_file, e))
        else:
            try:
                resources = json.load(sys.stdin)
            except (IOError, ValueError) as e:
                raise exceptions.BadFileException('Cannot read the standard input: {}'.format(e))
        return resources