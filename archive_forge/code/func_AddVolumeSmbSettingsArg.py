from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddVolumeSmbSettingsArg(parser):
    """Adds the --smb-settings arg to the arg parser."""
    parser.add_argument('--smb-settings', type=arg_parsers.ArgList(min_length=1, element_type=str), metavar='SMB_SETTING', help='List of settings specific to SMB protocol for a Cloud NetApp Files Volume. Valid component values are:\n  `ENCRYPT_DATA`, `BROWSABLE`, `CHANGE_NOTIFY`, `NON_BROWSABLE`,\n  `OPLOCKS`, `SHOW_SNAPSHOT`, `SHOW_PREVIOUS_VERSIONS`,\n  `ACCESS_BASED_ENUMERATION`, `CONTINUOUSLY_AVAILABLE`.')