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
def AddVolumeExportPolicyArg(parser):
    """Adds the Export Policy (--export-policy) arg to the given parser.

  Args:
    parser: argparse parser.
  """
    export_policy_arg_spec = {'allowed-clients': str, 'has-root-access': str, 'access-type': str, 'kerberos-5-read-only': arg_parsers.ArgBoolean(truthy_strings=netapp_util.truthy, falsey_strings=netapp_util.falsey), 'kerberos-5-read-write': arg_parsers.ArgBoolean(truthy_strings=netapp_util.truthy, falsey_strings=netapp_util.falsey), 'kerberos-5i-read-only': arg_parsers.ArgBoolean(truthy_strings=netapp_util.truthy, falsey_strings=netapp_util.falsey), 'kerberos-5i-read-write': arg_parsers.ArgBoolean(truthy_strings=netapp_util.truthy, falsey_strings=netapp_util.falsey), 'kerberos-5p-read-write': arg_parsers.ArgBoolean(truthy_strings=netapp_util.truthy, falsey_strings=netapp_util.falsey), 'kerberos-5p-read-only': arg_parsers.ArgBoolean(truthy_strings=netapp_util.truthy, falsey_strings=netapp_util.falsey), 'nfsv3': arg_parsers.ArgBoolean(truthy_strings=netapp_util.truthy, falsey_strings=netapp_util.falsey), 'nfsv4': arg_parsers.ArgBoolean(truthy_strings=netapp_util.truthy, falsey_strings=netapp_util.falsey)}
    export_policy_help = '        Export Policy of a Cloud NetApp Files Volume.\n        This will be a field similar to network\n        in which export policy fields can be specified as such:\n        `--export-policy=allowed-clients=ALLOWED_CLIENTS_IP_ADDRESSES,\n        has-root-access=HAS_ROOT_ACCESS_BOOL,access=ACCESS_TYPE,nfsv3=NFSV3,\n        nfsv4=NFSV4,kerberos-5-read-only=KERBEROS_5_READ_ONLY,\n        kerberos-5-read-write=KERBEROS_5_READ_WRITE,\n        kerberos-5i-read-only=KERBEROS_5I_READ_ONLY,\n        kerberos-5i-read-write=KERBEROS_5I_READ_WRITE,\n        kerberos-5p-read-only=KERBEROS_5P_READ_ONLY,\n        kerberos-5p-read-write=KERBEROS_5P_READ_WRITE`\n  '
    parser.add_argument('--export-policy', type=arg_parsers.ArgDict(spec=export_policy_arg_spec), action='append', help=export_policy_help)