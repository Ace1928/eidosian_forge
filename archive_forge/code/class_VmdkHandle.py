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
class VmdkHandle(FileHandle):
    """VMDK handle based on HttpNfcLease."""

    def __init__(self, session, lease, url, file_handle):
        self._session = session
        self._lease = lease
        self._url = url
        self._last_logged_progress = 0
        self._last_progress_udpate = 0
        super(VmdkHandle, self).__init__(file_handle)

    def _log_progress(self, progress):
        """Log data transfer progress."""
        if progress == 100 or progress - self._last_logged_progress >= MIN_PROGRESS_DIFF_TO_LOG:
            LOG.debug('Data transfer progress is %d%%.', progress)
            self._last_logged_progress = progress

    def _get_progress(self):
        """Get current progress for updating progress to lease."""
        pass

    def update_progress(self):
        """Updates progress to lease.

        This call back to the lease is essential to keep the lease alive
        across long running write/read operations.

        :raises: VimException, VimFaultException, VimAttributeException,
                 VimSessionOverLoadException, VimConnectionException
        """
        now = time.time()
        if now - self._last_progress_udpate < MIN_UPDATE_INTERVAL:
            return
        self._last_progress_udpate = now
        progress = int(self._get_progress())
        self._log_progress(progress)
        try:
            self._session.invoke_api(self._session.vim, 'HttpNfcLeaseProgress', self._lease, percent=progress)
        except exceptions.VimException:
            with excutils.save_and_reraise_exception():
                LOG.exception('Error occurred while updating the write/read progress of VMDK file with URL = %s.', self._url)

    def _release_lease(self):
        """Release the lease

        :raises: VimException, VimFaultException, VimAttributeException,
                 VimSessionOverLoadException, VimConnectionException
        """
        LOG.debug('Getting lease state for %s.', self._url)
        state = self._session.invoke_api(vim_util, 'get_object_property', self._session.vim, self._lease, 'state')
        LOG.debug('Lease for %(url)s is in state: %(state)s.', {'url': self._url, 'state': state})
        if self._get_progress() < 100:
            LOG.error('Aborting lease for %s due to incomplete transfer.', self._url)
            self._session.invoke_api(self._session.vim, 'HttpNfcLeaseAbort', self._lease)
        elif state == 'ready':
            LOG.debug('Releasing lease for %s.', self._url)
            self._session.invoke_api(self._session.vim, 'HttpNfcLeaseComplete', self._lease)
        else:
            LOG.debug('Lease for %(url)s is in state: %(state)s; no need to release.', {'url': self._url, 'state': state})

    @staticmethod
    def _create_import_vapp_lease(session, rp_ref, import_spec, vm_folder_ref):
        """Create and wait for HttpNfcLease lease for vApp import."""
        LOG.debug('Creating HttpNfcLease lease for vApp import into resource pool: %s.', rp_ref)
        lease = session.invoke_api(session.vim, 'ImportVApp', rp_ref, spec=import_spec, folder=vm_folder_ref)
        LOG.debug('Lease: %(lease)s obtained for vApp import into resource pool %(rp_ref)s.', {'lease': lease, 'rp_ref': rp_ref})
        session.wait_for_lease_ready(lease)
        LOG.debug('Invoking VIM API for reading info of lease: %s.', lease)
        lease_info = session.invoke_api(vim_util, 'get_object_property', session.vim, lease, 'info')
        return (lease, lease_info)

    @staticmethod
    def _create_export_vm_lease(session, vm_ref):
        """Create and wait for HttpNfcLease lease for VM export."""
        LOG.debug('Creating HttpNfcLease lease for exporting VM: %s.', vm_ref)
        lease = session.invoke_api(session.vim, 'ExportVm', vm_ref)
        LOG.debug('Lease: %(lease)s obtained for exporting VM: %(vm_ref)s.', {'lease': lease, 'vm_ref': vm_ref})
        session.wait_for_lease_ready(lease)
        LOG.debug('Invoking VIM API for reading info of lease: %s.', lease)
        lease_info = session.invoke_api(vim_util, 'get_object_property', session.vim, lease, 'info')
        return (lease, lease_info)

    @staticmethod
    def _fix_esx_url(url, host, port):
        """Fix netloc in the case of an ESX host.

        In the case of an ESX host, the netloc is set to '*' in the URL
        returned in HttpNfcLeaseInfo. It should be replaced with host name
        or IP address.
        """
        urlp = urlparse.urlparse(url)
        if urlp.netloc == '*':
            scheme, netloc, path, params, query, fragment = urlp
            if netutils.is_valid_ipv6(host):
                netloc = '[%s]:%d' % (host, port)
            else:
                netloc = '%s:%d' % (host, port)
            url = urlparse.urlunparse((scheme, netloc, path, params, query, fragment))
        return url

    @staticmethod
    def _find_vmdk_url(lease_info, host, port):
        """Find the URL corresponding to a VMDK file in lease info."""
        url = None
        ssl_thumbprint = None
        for deviceUrl in lease_info.deviceUrl:
            if deviceUrl.disk:
                url = VmdkHandle._fix_esx_url(deviceUrl.url, host, port)
                ssl_thumbprint = deviceUrl.sslThumbprint
                break
        if not url:
            excep_msg = _('Could not retrieve VMDK URL from lease info.')
            LOG.error(excep_msg)
            raise exceptions.VimException(excep_msg)
        LOG.debug('Found VMDK URL: %s from lease info.', url)
        return (url, ssl_thumbprint)