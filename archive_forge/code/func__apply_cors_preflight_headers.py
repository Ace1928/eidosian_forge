import copy
import logging
import debtcollector
from oslo_config import cfg
from oslo_middleware import base
import webob.exc
def _apply_cors_preflight_headers(self, request, response):
    """Handle CORS Preflight (Section 6.2)

        Given a request and a response, apply the CORS preflight headers
        appropriate for the request.
        """
    if 200 > response.status_code or response.status_code >= 300:
        response = base.NoContentTypeResponse(status=webob.exc.HTTPOk.code)
    if 'Origin' not in request.headers:
        return response
    try:
        origin, cors_config = self._get_cors_config_by_origin(request.headers['Origin'])
    except InvalidOriginError:
        return response
    if 'Access-Control-Request-Method' not in request.headers:
        LOG.debug('CORS request does not contain Access-Control-Request-Method header.')
        return response
    request_method = request.headers['Access-Control-Request-Method']
    try:
        request_headers = self._split_header_values(request, 'Access-Control-Request-Headers')
    except Exception:
        LOG.debug('Cannot parse request headers.')
        return response
    permitted_methods = cors_config['allow_methods']
    if request_method not in permitted_methods:
        LOG.debug("Request method '%s' not in permitted list: %s" % (request_method, permitted_methods))
        return response
    permitted_headers = [header.upper() for header in cors_config['allow_headers'] + self.simple_headers]
    for requested_header in request_headers:
        upper_header = requested_header.upper()
        if upper_header not in permitted_headers:
            LOG.debug("Request header '%s' not in permitted list: %s" % (requested_header, permitted_headers))
            return response
    response.headers['Vary'] = 'Origin'
    response.headers['Access-Control-Allow-Origin'] = origin
    if cors_config['allow_credentials']:
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    if 'max_age' in cors_config and cors_config['max_age']:
        response.headers['Access-Control-Max-Age'] = str(cors_config['max_age'])
    response.headers['Access-Control-Allow-Methods'] = request_method
    if request_headers:
        response.headers['Access-Control-Allow-Headers'] = ','.join(request_headers)
    return response