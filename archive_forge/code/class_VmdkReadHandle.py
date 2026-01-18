import logging
import ssl
import time
from oslo_utils import excutils
from oslo_utils import netutils
import requests
import urllib.parse as urlparse
from urllib3 import connection as httplib
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware import vim_util
class VmdkReadHandle(VmdkHandle):
    """VMDK read handle based on HttpNfcLease."""

    def __init__(self, session, host, port, vm_ref, vmdk_path, vmdk_size):
        """Initializes the VMDK read handle with the given parameters.

        During the read (export) operation, the VMDK file is converted to a
        stream-optimized sparse disk format. Therefore, the size of the VMDK
        file read may be smaller than the actual VMDK size.

        :param session: valid api session to ESX/VC server
        :param host: ESX/VC server IP address or host name
        :param port: port for connection
        :param vm_ref: managed object reference of the backing VM whose VMDK
                       is to be exported
        :param vmdk_path: path of the VMDK file to be exported
        :param vmdk_size: actual size of the VMDK file
        :raises: VimException, VimFaultException, VimAttributeException,
                 VimSessionOverLoadException, VimConnectionException
        """
        self._vmdk_size = vmdk_size
        self._bytes_read = 0
        lease, lease_info = self._create_export_vm_lease(session, vm_ref)
        url, thumbprint = self._find_vmdk_url(lease_info, host, port)
        cookies = session.vim.client.cookiejar
        self._conn = self._create_read_connection(url, cookies=cookies, ssl_thumbprint=thumbprint)
        super(VmdkReadHandle, self).__init__(session, lease, url, self._conn.getresponse())

    def read(self, chunk_size=READ_CHUNKSIZE):
        """Read a chunk of data from the VMDK file.

        :param chunk_size: size of read chunk
        :returns: the data
        :raises: VimException
        """
        try:
            data = self._file_handle.read(chunk_size)
            self._bytes_read += len(data)
            return data
        except Exception as excep:
            excep_msg = _('Error occurred while reading data from %s.') % self._url
            LOG.exception(excep_msg)
            raise exceptions.VimException(excep_msg, excep)

    def tell(self):
        return self._bytes_read

    def close(self):
        """Releases the lease and close the connection.

        :raises: VimException, VimFaultException, VimAttributeException,
                 VimSessionOverLoadException, VimConnectionException
        """
        self._conn.close()
        try:
            self._release_lease()
        except exceptions.ManagedObjectNotFoundException:
            LOG.info('Lease for %(url)s not found.  No need to release.', {'url': self._url})
            return
        except exceptions.VimException:
            LOG.warning('Error occurred while releasing the lease for %s.', self._url, exc_info=True)
            raise
        finally:
            super(VmdkReadHandle, self).close()
        LOG.debug('Closed VMDK read handle for %s.', self._url)

    def _get_progress(self):
        return float(self._bytes_read) / self._vmdk_size * 100

    def __str__(self):
        return 'VMDK read handle for %s' % self._url