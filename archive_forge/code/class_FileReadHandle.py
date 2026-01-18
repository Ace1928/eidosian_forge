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
class FileReadHandle(FileHandle):
    """Read handle for a file in VMware server."""

    def __init__(self, host_or_url, port=None, data_center_name=None, datastore_name=None, cookies=None, file_path=None, scheme='https', cacerts=False, thumbprint=None):
        """Initializes the read handle with given parameters.

        :param host_or_url: ESX/VC server IP address or host name or a complete
                            DatastoreURL
        :param port: port for connection
        :param data_center_name: name of the data center in the case of a VC
                                 server
        :param datastore_name: name of the datastore where the file is stored
        :param cookies: cookies to build the vim cookie header, or a string
                        with the prebuild vim cookie header
                        (See: DatastoreURL.get_transfer_ticket())
        :param file_path: datastore path where the file is written
        :param scheme: protocol-- http or https
        :param cacerts: CA bundle file to use for SSL verification
        :param thumbprint: expected SHA1 thumbprint of server's certificate
        :raises: VimConnectionException, ValueError
        """
        if not port and (not data_center_name) and (not datastore_name):
            self._url = host_or_url
        else:
            soap_url = self._get_soap_url(scheme, host_or_url, port)
            param_list = {'dcPath': data_center_name, 'dsName': datastore_name}
            self._url = '%s/folder/%s' % (soap_url, file_path)
            self._url = self._url + '?' + urlparse.urlencode(param_list)
        self._conn = self._create_read_connection(self._url, cookies=cookies, cacerts=cacerts, ssl_thumbprint=thumbprint)
        FileHandle.__init__(self, self._conn.getresponse())

    def read(self, length):
        """Read data from the file.
        :param length: amount of data to be read
        :raises: VimConnectionException, VimException
        """
        try:
            return self._file_handle.read(length)
        except requests.RequestException as excep:
            excep_msg = _('Connection error occurred while reading data from %s.') % self._url
            LOG.exception(excep_msg)
            raise exceptions.VimConnectionException(excep_msg, excep)
        except Exception as excep:
            excep_msg = _('Error occurred while writing data to %s.') % self._url
            LOG.exception(excep_msg)
            raise exceptions.VimException(excep_msg, excep)

    def close(self):
        """Closes the connection.
        """
        self._conn.close()
        super(FileReadHandle, self).close()
        LOG.debug('Closed File read handle for %s.', self._url)

    def get_size(self):
        return self._file_handle.getheader('Content-Length')

    def __str__(self):
        return 'File read handle for %s' % self._url