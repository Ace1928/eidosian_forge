import http.client as http
import os
import sys
import urllib.parse as urlparse
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import uuidutils
from webob import exc
from glance.common import config
from glance.common import exception
from glance.common import utils
from glance.i18n import _, _LE, _LI, _LW
class ImageService(object):

    def __init__(self, conn, auth_token):
        """Initialize the ImageService.

        :param conn: a http.client.HTTPConnection to the glance server
        :param auth_token: authentication token to pass in the x-auth-token
            header
        """
        self.auth_token = auth_token
        self.conn = conn

    def _http_request(self, method, url, headers, body, ignore_result_body=False):
        """Perform an HTTP request against the server.

        method: the HTTP method to use
        url: the URL to request (not including server portion)
        headers: headers for the request
        body: body to send with the request
        ignore_result_body: the body of the result will be ignored

        :returns: A http.client response object
        """
        if self.auth_token:
            headers.setdefault('x-auth-token', self.auth_token)
        LOG.debug('Request: %(method)s http://%(server)s:%(port)s%(url)s with headers %(headers)s', {'method': method, 'server': self.conn.host, 'port': self.conn.port, 'url': url, 'headers': repr(headers)})
        self.conn.request(method, url, body, headers)
        response = self.conn.getresponse()
        headers = self._header_list_to_dict(response.getheaders())
        code = response.status
        code_description = http.responses[code]
        LOG.debug('Response: %(code)s %(status)s %(headers)s', {'code': code, 'status': code_description, 'headers': repr(headers)})
        if code == http.BAD_REQUEST:
            raise exc.HTTPBadRequest(explanation=response.read())
        if code == http.INTERNAL_SERVER_ERROR:
            raise exc.HTTPInternalServerError(explanation=response.read())
        if code == http.UNAUTHORIZED:
            raise exc.HTTPUnauthorized(explanation=response.read())
        if code == http.FORBIDDEN:
            raise exc.HTTPForbidden(explanation=response.read())
        if code == http.CONFLICT:
            raise exc.HTTPConflict(explanation=response.read())
        if ignore_result_body:
            response.read()
        return response

    def get_images(self):
        """Return a detailed list of images.

        Yields a series of images as dicts containing metadata.
        """
        params = {'is_public': None}
        while True:
            url = '/v1/images/detail'
            query = urlparse.urlencode(params)
            if query:
                url += '?%s' % query
            response = self._http_request('GET', url, {}, '')
            result = jsonutils.loads(response.read())
            if not result or 'images' not in result or (not result['images']):
                return
            for image in result.get('images', []):
                params['marker'] = image['id']
                yield image

    def get_image(self, image_uuid):
        """Fetch image data from glance.

        image_uuid: the id of an image

        :returns: a http.client Response object where the body is the image.
        """
        url = '/v1/images/%s' % image_uuid
        return self._http_request('GET', url, {}, '')

    @staticmethod
    def _header_list_to_dict(headers):
        """Expand a list of headers into a dictionary.

        headers: a list of [(key, value), (key, value), (key, value)]

        Returns: a dictionary representation of the list
        """
        d = {}
        for header, value in headers:
            if header.startswith('x-image-meta-property-'):
                prop = header.replace('x-image-meta-property-', '')
                d.setdefault('properties', {})
                d['properties'][prop] = value
            else:
                d[header.replace('x-image-meta-', '')] = value
        return d

    def get_image_meta(self, image_uuid):
        """Return the metadata for a single image.

        image_uuid: the id of an image

        Returns: image metadata as a dictionary
        """
        url = '/v1/images/%s' % image_uuid
        response = self._http_request('HEAD', url, {}, '', ignore_result_body=True)
        return self._header_list_to_dict(response.getheaders())

    @staticmethod
    def _dict_to_headers(d):
        """Convert a dictionary into one suitable for a HTTP request.

        d: a dictionary

        Returns: the same dictionary, with x-image-meta added to every key
        """
        h = {}
        for key in d:
            if key == 'properties':
                for subkey in d[key]:
                    if d[key][subkey] is None:
                        h['x-image-meta-property-%s' % subkey] = ''
                    else:
                        h['x-image-meta-property-%s' % subkey] = d[key][subkey]
            else:
                h['x-image-meta-%s' % key] = d[key]
        return h

    def add_image(self, image_meta, image_data):
        """Upload an image.

        image_meta: image metadata as a dictionary
        image_data: image data as a object with a read() method

        Returns: a tuple of (http response headers, http response body)
        """
        url = '/v1/images'
        headers = self._dict_to_headers(image_meta)
        headers['Content-Type'] = 'application/octet-stream'
        headers['Content-Length'] = int(image_meta['size'])
        response = self._http_request('POST', url, headers, image_data)
        headers = self._header_list_to_dict(response.getheaders())
        LOG.debug('Image post done')
        body = response.read()
        return (headers, body)

    def add_image_meta(self, image_meta):
        """Update image metadata.

        image_meta: image metadata as a dictionary

        Returns: a tuple of (http response headers, http response body)
        """
        url = '/v1/images/%s' % image_meta['id']
        headers = self._dict_to_headers(image_meta)
        headers['Content-Type'] = 'application/octet-stream'
        response = self._http_request('PUT', url, headers, '')
        headers = self._header_list_to_dict(response.getheaders())
        LOG.debug('Image post done')
        body = response.read()
        return (headers, body)