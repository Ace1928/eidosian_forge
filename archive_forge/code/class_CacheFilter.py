import http.client as http
import re
from oslo_log import log as logging
import webob
from glance.api.common import size_checked_iter
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
import glance.db
from glance.i18n import _LE, _LI
from glance import image_cache
from glance import notifier
class CacheFilter(wsgi.Middleware):

    def __init__(self, app):
        self.cache = image_cache.ImageCache()
        self.policy = policy.Enforcer()
        LOG.info(_LI('Initialized image cache middleware'))
        super(CacheFilter, self).__init__(app)

    def _verify_metadata(self, image_meta):
        """
        Sanity check the 'deleted' and 'size' metadata values.
        """
        if image_meta['status'] == 'deleted' and image_meta['deleted']:
            raise exception.NotFound()
        if not image_meta['size']:
            if not isinstance(image_meta, policy.ImageTarget):
                image_meta['size'] = self.cache.get_image_size(image_meta['id'])
            else:
                image_meta.target.size = self.cache.get_image_size(image_meta['id'])

    @staticmethod
    def _match_request(request):
        """Determine the version of the url and extract the image id

        :returns: tuple of version and image id if the url is a cacheable,
                 otherwise None
        """
        for (version, method), pattern in PATTERNS.items():
            if request.method != method:
                continue
            match = pattern.match(request.path_info)
            if match is None:
                continue
            image_id = match.group(1)
            if image_id != 'detail':
                return (version, method, image_id)

    def _enforce(self, req, image):
        """Authorize an action against our policies"""
        api_pol = api_policy.ImageAPIPolicy(req.context, image, self.policy)
        api_pol.download_image()

    def _get_v2_image_metadata(self, request, image_id):
        """
        Retrieves image and for v2 api and creates adapter like object
        to access image core or custom properties on request.
        """
        db_api = glance.db.get_api()
        image_repo = glance.db.ImageRepo(request.context, db_api)
        try:
            image = image_repo.get(image_id)
            request.environ['api.cache.image'] = image
            return (image, policy.ImageTarget(image))
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg, request=request)

    def process_request(self, request):
        """
        For requests for an image file, we check the local image
        cache. If present, we return the image file, appending
        the image metadata in headers. If not present, we pass
        the request on to the next application in the pipeline.
        """
        match = self._match_request(request)
        try:
            version, method, image_id = match
        except TypeError:
            return None
        self._stash_request_info(request, image_id, method, version)
        if request.headers.get('Content-Range') or request.headers.get('Range'):
            return None
        if request.method != 'GET' or not self.cache.is_cached(image_id):
            return None
        method = getattr(self, '_get_%s_image_metadata' % version)
        image, metadata = method(request, image_id)
        if metadata['status'] == 'deactivated':
            return None
        self._enforce(request, image)
        LOG.debug("Cache hit for image '%s'", image_id)
        image_iterator = self.get_from_cache(image_id)
        method = getattr(self, '_process_%s_request' % version)
        try:
            return method(request, image_id, image_iterator, metadata)
        except exception.ImageNotFound:
            msg = _LE("Image cache contained image file for image '%s', however the database did not contain metadata for that image!") % image_id
            LOG.error(msg)
            self.cache.delete_cached_image(image_id)

    @staticmethod
    def _stash_request_info(request, image_id, method, version):
        """
        Preserve the image id, version and request method for later retrieval
        """
        request.environ['api.cache.image_id'] = image_id
        request.environ['api.cache.method'] = method
        request.environ['api.cache.version'] = version

    @staticmethod
    def _fetch_request_info(request):
        """
        Preserve the cached image id, version for consumption by the
        process_response method of this middleware
        """
        try:
            image_id = request.environ['api.cache.image_id']
            method = request.environ['api.cache.method']
            version = request.environ['api.cache.version']
        except KeyError:
            return None
        else:
            return (image_id, method, version)

    def _process_v2_request(self, request, image_id, image_iterator, image_meta):
        image = request.environ['api.cache.image']
        self._verify_metadata(image_meta)
        response = webob.Response(request=request)
        response.app_iter = size_checked_iter(response, image_meta, image_meta['size'], image_iterator, notifier.Notifier())
        response.headers['Content-Type'] = 'application/octet-stream'
        if image.checksum:
            response.headers['Content-MD5'] = image.checksum
        response.headers['Content-Length'] = str(image.size)
        return response

    def process_response(self, resp):
        """
        We intercept the response coming back from the main
        images Resource, removing image file from the cache
        if necessary
        """
        status_code = self.get_status_code(resp)
        if not 200 <= status_code < 300:
            return resp
        if status_code == http.PARTIAL_CONTENT:
            return resp
        try:
            image_id, method, version = self._fetch_request_info(resp.request)
        except TypeError:
            return resp
        if method == 'GET' and status_code == http.NO_CONTENT:
            return resp
        method_str = '_process_%s_response' % method
        try:
            process_response_method = getattr(self, method_str)
        except AttributeError:
            LOG.error(_LE('could not find %s'), method_str)
            return resp
        else:
            return process_response_method(resp, image_id, version=version)

    def _process_DELETE_response(self, resp, image_id, version=None):
        if self.cache.is_cached(image_id):
            LOG.debug('Removing image %s from cache', image_id)
            self.cache.delete_cached_image(image_id)
        return resp

    def _process_GET_response(self, resp, image_id, version=None):
        image_checksum = resp.headers.get('Content-MD5')
        if not image_checksum:
            image_checksum = resp.headers.get('x-image-meta-checksum')
        if not image_checksum:
            LOG.error(_LE('Checksum header is missing.'))
        image = None
        if version:
            method = getattr(self, '_get_%s_image_metadata' % version)
            image, metadata = method(resp.request, image_id)
        self._enforce(resp.request, image)
        resp.app_iter = self.cache.get_caching_iter(image_id, image_checksum, resp.app_iter)
        return resp

    def get_status_code(self, response):
        """
        Returns the integer status code from the response, which
        can be either a Webob.Response (used in testing) or httplib.Response
        """
        if hasattr(response, 'status_int'):
            return response.status_int
        return response.status

    def get_from_cache(self, image_id):
        """Called if cache hit"""
        with self.cache.open_for_read(image_id) as cache_file:
            chunks = utils.chunkiter(cache_file)
            for chunk in chunks:
                yield chunk