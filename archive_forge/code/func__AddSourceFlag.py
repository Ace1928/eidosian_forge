from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.console import console_io
def _AddSourceFlag(parser, schema_path=None):
    help_text = 'Path to a YAML file containing configuration export data.\n        The YAML file must not contain any output-only fields. Alternatively,\n        you may omit this flag to read from standard input. For a schema\n        describing the export/import format, see: {}.\n      '.format(schema_path)
    parser.add_argument('--source', help=textwrap.dedent(help_text), required=False)