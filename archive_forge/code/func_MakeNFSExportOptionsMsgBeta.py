from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.filestore.backups import util as backup_util
from googlecloudsdk.command_lib.filestore.snapshots import util as snapshot_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
@staticmethod
def MakeNFSExportOptionsMsgBeta(messages, nfs_export_options):
    """Creates an MakeNFSExportOptionsMsgBeta message.

    This function is a copy MakeNFSExportOptionsMsg with addition of handling
    the SecurityFlavors for NFSv41.

    Args:
      messages: The messages module.
      nfs_export_options: list, containing NfsExportOptions dictionary.

    Returns:
      File share message populated with values, filled with defaults.
      In case no nfs export options are provided we rely on the API to apply a
      default.
    """
    read_write = 'READ_WRITE'
    root_squash = 'ROOT_SQUASH'
    no_root_squash = 'NO_ROOT_SQUASH'
    anonimous_uid = 65534
    anonimous_gid = 65534
    nfs_export_configs = []
    if nfs_export_options is None:
        return []
    for nfs_export_option in nfs_export_options:
        access_mode = messages.NfsExportOptions.AccessModeValueValuesEnum.lookup_by_name(nfs_export_option.get('access-mode', read_write))
        squash_mode = messages.NfsExportOptions.SquashModeValueValuesEnum.lookup_by_name(nfs_export_option.get('squash-mode', no_root_squash))
        if nfs_export_option.get('squash-mode', None) == root_squash:
            anon_uid = nfs_export_option.get('anon_uid', anonimous_uid)
            anon_gid = nfs_export_option.get('anon_gid', anonimous_gid)
        else:
            anon_gid = None
            anon_uid = None
        security_flavors_list = []
        flavors = nfs_export_option.get('security-flavors', [])
        for flavor in flavors:
            security_flavors_list.append(messages.NfsExportOptions.SecurityFlavorsValueListEntryValuesEnum.lookup_by_name(flavor))
        nfs_export_config = messages.NfsExportOptions(ipRanges=nfs_export_option.get('ip-ranges', []), anonUid=anon_uid, anonGid=anon_gid, accessMode=access_mode, squashMode=squash_mode, securityFlavors=security_flavors_list)
        nfs_export_configs.append(nfs_export_config)
    return nfs_export_configs