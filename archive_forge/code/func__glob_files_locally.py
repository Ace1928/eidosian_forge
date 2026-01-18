from __future__ import absolute_import, division, print_function
import os
import mimetypes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def _glob_files_locally(folder_path):
    len_folder_path = len(folder_path) + 1
    for root, v, files in os.walk(folder_path):
        for f in files:
            full_path = os.path.join(root, f)
            yield (full_path, full_path[len_folder_path:])