import logging
import posixpath
import random
import re
import http.client as httplib
import urllib.parse as urlparse
from oslo_vmware._i18n import _
from oslo_vmware import constants
from oslo_vmware import exceptions
from oslo_vmware import vim_util
class DatastoreURL(object):
    """Class for representing a URL to HTTP access a file in a datastore.

    This provides various helper methods to access components and useful
    variants of the datastore URL.
    """

    def __init__(self, scheme, server, path, datacenter_path, datastore_name):
        self._scheme = scheme
        self._server = server
        self._path = path
        self._datacenter_path = datacenter_path
        self._datastore_name = datastore_name
        params = {'dcPath': self._datacenter_path, 'dsName': self._datastore_name}
        self._query = urlparse.urlencode(params)

    @classmethod
    def urlparse(cls, url):
        scheme, server, path, params, query, fragment = urlparse.urlparse(url)
        if not query:
            path = path.split('?')
            query = path[1]
            path = path[0]
        params = urlparse.parse_qs(query)
        dc_path = params.get('dcPath')
        if dc_path is not None and len(dc_path) > 0:
            datacenter_path = dc_path[0]
        ds_name = params.get('dsName')
        if ds_name is not None and len(ds_name) > 0:
            datastore_name = ds_name[0]
        path = path[len('/folder'):]
        return cls(scheme, server, path, datacenter_path, datastore_name)

    @property
    def path(self):
        return self._path.strip('/')

    @property
    def datacenter_path(self):
        return self._datacenter_path

    @property
    def datastore_name(self):
        return self._datastore_name

    def __str__(self):
        return '%s://%s/folder/%s?%s' % (self._scheme, self._server, self.path, self._query)

    def connect(self, method, content_length, cookie):
        try:
            if self._scheme == 'http':
                conn = httplib.HTTPConnection(self._server)
            elif self._scheme == 'https':
                conn = httplib.HTTPSConnection(self._server)
            else:
                excep_msg = _('Invalid scheme: %s.') % self._scheme
                LOG.error(excep_msg)
                raise ValueError(excep_msg)
            conn.putrequest(method, '/folder/%s?%s' % (self.path, self._query))
            conn.putheader('User-Agent', constants.USER_AGENT)
            conn.putheader('Content-Length', content_length)
            conn.putheader('Cookie', cookie)
            conn.endheaders()
            LOG.debug('Created HTTP connection to transfer the file with URL = %s.', str(self))
            return conn
        except (httplib.InvalidURL, httplib.CannotSendRequest, httplib.CannotSendHeader) as excep:
            excep_msg = _('Error occurred while creating HTTP connection to write to file with URL = %s.') % str(self)
            LOG.exception(excep_msg)
            raise exceptions.VimConnectionException(excep_msg, excep)

    def get_transfer_ticket(self, session, method):
        client_factory = session.vim.client.factory
        spec = vim_util.get_http_service_request_spec(client_factory, method, str(self))
        ticket = session.invoke_api(session.vim, 'AcquireGenericServiceTicket', session.vim.service_content.sessionManager, spec=spec)
        return '%s="%s"' % (constants.CGI_COOKIE_KEY, ticket.id)