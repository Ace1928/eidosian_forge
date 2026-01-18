from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import migrate_config
class DispatchXmlToYaml(base.Command):
    """Convert a dispatch.xml file to dispatch.yaml."""

    @staticmethod
    def Args(parser):
        parser.add_argument('xml_file', help='Path to the dispatch.xml file.')

    def Run(self, args):
        src = os.path.abspath(args.xml_file)
        dst = os.path.join(os.path.dirname(src), 'dispatch.yaml')
        entry = migrate_config.REGISTRY['dispatch-xml-to-yaml']
        migrate_config.Run(entry, src=src, dst=dst)