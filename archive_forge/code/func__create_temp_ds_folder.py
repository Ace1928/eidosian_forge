import os
import tempfile
from oslo_log import log as logging
from oslo_utils import fileutils
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator import initiator_connector
def _create_temp_ds_folder(self, session, ds_folder_path, dc_ref):
    fileManager = session.vim.service_content.fileManager
    try:
        session.invoke_api(session.vim, 'MakeDirectory', fileManager, name=ds_folder_path, datacenter=dc_ref)
    except oslo_vmw_exceptions.FileAlreadyExistsException:
        pass