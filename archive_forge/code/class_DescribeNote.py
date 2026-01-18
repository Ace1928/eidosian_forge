from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.containeranalysis import util as containeranalysis_util
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
class DescribeNote(base.DescribeCommand):
    """Describe a Google note."""

    @staticmethod
    def Args(parser):
        parser.add_argument('note_name', help='Name, relative name, or URL of the note.')

    def Run(self, args):
        return containeranalysis_util.MakeGetNoteRequest(args.note_name, properties.VALUES.core.project.Get(required=True))