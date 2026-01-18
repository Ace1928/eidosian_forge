from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
def AddMetadataArgs(parser, required=False):
    """Adds --metadata and --metadata-from-file flags."""
    metadata_help = "      Metadata to be made available to the guest operating system\n      running on the instances. Each metadata entry is a key/value\n      pair separated by an equals sign. Each metadata key must be unique\n      and have a max of 128 bytes in length. Each value must have a max of\n      256 KB in length. Multiple arguments can be\n      passed to this flag, e.g.,\n      ``--metadata key-1=value-1,key-2=value-2,key-3=value-3''.\n      The combined total size for all metadata entries is 512 KB.\n\n      In images that have Compute Engine tools installed on them,\n      such as the\n      link:https://cloud.google.com/compute/docs/images[official images],\n      the following metadata keys have special meanings:\n\n      *startup-script*::: Specifies a script that will be executed\n      by the instances once they start running. For convenience,\n      ``--metadata-from-file'' can be used to pull the value from a\n      file.\n\n      *startup-script-url*::: Same as ``startup-script'' except that\n      the script contents are pulled from a publicly-accessible\n      location on the web.\n\n\n      For startup scripts on Windows instances, the following metadata keys\n      have special meanings:\n      ``windows-startup-script-url'',\n      ``windows-startup-script-cmd'', ``windows-startup-script-bat'',\n      ``windows-startup-script-ps1'', ``sysprep-specialize-script-url'',\n      ``sysprep-specialize-script-cmd'', ``sysprep-specialize-script-bat'',\n      and ``sysprep-specialize-script-ps1''. For more information, see\n      [Running startup scripts](https://cloud.google.com/compute/docs/startupscript).\n      "
    if required:
        metadata_help += '\n\n      At least one of [--metadata] or [--metadata-from-file] is required.\n      '
    parser.add_argument('--metadata', type=arg_parsers.ArgDict(min_length=1), default={}, help=metadata_help, metavar='KEY=VALUE', action=arg_parsers.StoreOnceAction)
    metadata_from_file_help = "      Same as ``--metadata'' except that the value for the entry will\n      be read from a local file. This is useful for values that are\n      too large such as ``startup-script'' contents.\n      "
    if required:
        metadata_from_file_help += '\n\n      At least one of [--metadata] or [--metadata-from-file] is required.\n      '
    parser.add_argument('--metadata-from-file', type=arg_parsers.ArgDict(min_length=1), default={}, help=metadata_from_file_help, metavar='KEY=LOCAL_FILE_PATH')