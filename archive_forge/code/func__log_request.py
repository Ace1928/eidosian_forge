from __future__ import absolute_import
import six
import json
import logging
import platform
from six.moves.urllib.parse import urlencode
from googleapiclient.errors import HttpError
def _log_request(self, headers, path_params, query, body):
    """Logs debugging information about the request if requested."""
    if dump_request_response:
        LOGGER.info('--request-start--')
        LOGGER.info('-headers-start-')
        for h, v in six.iteritems(headers):
            LOGGER.info('%s: %s', h, v)
        LOGGER.info('-headers-end-')
        LOGGER.info('-path-parameters-start-')
        for h, v in six.iteritems(path_params):
            LOGGER.info('%s: %s', h, v)
        LOGGER.info('-path-parameters-end-')
        LOGGER.info('body: %s', body)
        LOGGER.info('query: %s', query)
        LOGGER.info('--request-end--')