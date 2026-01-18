import errno
import hashlib
import os.path  # pylint: disable-msg=W0404
import warnings
from typing import Dict, List, Type, Iterator, Optional
from os.path import join as pjoin
import libcloud.utils.files
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import BaseDriver, Connection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError
from libcloud.storage.types import ObjectDoesNotExistError
def _save_object(self, response, obj, destination_path, overwrite_existing=False, delete_on_failure=True, chunk_size=None, partial_download=False):
    """
        Save object to the provided path.

        :param response: RawResponse instance.
        :type response: :class:`RawResponse`

        :param obj: Object instance.
        :type obj: :class:`Object`

        :param destination_path: Destination directory.
        :type destination_path: ``str``

        :param delete_on_failure: True to delete partially downloaded object if
                                  the download fails.
        :type delete_on_failure: ``bool``

        :param overwrite_existing: True to overwrite a local path if it already
                                   exists.
        :type overwrite_existing: ``bool``

        :param chunk_size: Optional chunk size
            (defaults to ``libcloud.storage.base.CHUNK_SIZE``, 8kb)
        :type chunk_size: ``int``

        :param partial_download: True if this is a range (partial) save,
                                 False otherwise.
        :type partial_download: ``bool``

        :return: ``True`` on success, ``False`` otherwise.
        :rtype: ``bool``
        """
    chunk_size = chunk_size or CHUNK_SIZE
    base_name = os.path.basename(destination_path)
    if not base_name and (not os.path.exists(destination_path)):
        raise LibcloudError(value='Path %s does not exist' % destination_path, driver=self)
    if not base_name:
        file_path = pjoin(destination_path, obj.name)
    else:
        file_path = destination_path
    if os.path.exists(file_path) and (not overwrite_existing):
        raise LibcloudError(value='File %s already exists, but ' % file_path + 'overwrite_existing=False', driver=self)
    bytes_transferred = 0
    with open(file_path, 'wb') as file_handle:
        for chunk in response._response.iter_content(chunk_size):
            file_handle.write(b(chunk))
            bytes_transferred += len(chunk)
    if not partial_download and int(obj.size) != int(bytes_transferred):
        if delete_on_failure:
            try:
                os.unlink(file_path)
            except Exception:
                pass
        return False
    return True