import os
import ssl
import copy
import json
import time
import socket
import binascii
from typing import Any, Dict, Type, Union, Optional
import libcloud
from libcloud.http import LibcloudConnection, HttpLibResponseProxy
from libcloud.utils.py3 import ET, httplib, urlparse, urlencode
from libcloud.utils.misc import lowercase_keys
from libcloud.utils.retry import Retry
from libcloud.common.types import LibcloudError, MalformedResponseError
from libcloud.common.exceptions import exception_from_message
class PollingConnection(Connection):
    """
    Connection class which can also work with the async APIs.

    After initial requests, this class periodically polls for jobs status and
    waits until the job has finished.
    If job doesn't finish in timeout seconds, an Exception thrown.
    """
    poll_interval = 0.5
    timeout = 200
    request_method = 'request'

    def async_request(self, action, params=None, data=None, headers=None, method='GET', context=None):
        """
        Perform an 'async' request to the specified path. Keep in mind that
        this function is *blocking* and 'async' in this case means that the
        hit URL only returns a job ID which is the periodically polled until
        the job has completed.

        This function works like this:

        - Perform a request to the specified path. Response should contain a
          'job_id'.

        - Returned 'job_id' is then used to construct a URL which is used for
          retrieving job status. Constructed URL is then periodically polled
          until the response indicates that the job has completed or the
          timeout of 'self.timeout' seconds has been reached.

        :type action: ``str``
        :param action: A path

        :type params: ``dict``
        :param params: Optional mapping of additional parameters to send. If
            None, leave as an empty ``dict``.

        :type data: ``unicode``
        :param data: A body of data to send with the request.

        :type headers: ``dict``
        :param headers: Extra headers to add to the request
            None, leave as an empty ``dict``.

        :type method: ``str``
        :param method: An HTTP method such as "GET" or "POST".

        :type context: ``dict``
        :param context: Context dictionary which is passed to the functions
                        which construct initial and poll URL.

        :return: An :class:`Response` instance.
        :rtype: :class:`Response` instance
        """
        request = getattr(self, self.request_method)
        kwargs = self.get_request_kwargs(action=action, params=params, data=data, headers=headers, method=method, context=context)
        response = request(**kwargs)
        kwargs = self.get_poll_request_kwargs(response=response, context=context, request_kwargs=kwargs)
        end = time.time() + self.timeout
        completed = False
        while time.time() < end and (not completed):
            response = request(**kwargs)
            completed = self.has_completed(response=response)
            if not completed:
                time.sleep(self.poll_interval)
        if not completed:
            raise LibcloudError('Job did not complete in %s seconds' % self.timeout)
        return response

    def get_request_kwargs(self, action, params=None, data=None, headers=None, method='GET', context=None):
        """
        Arguments which are passed to the initial request() call inside
        async_request.
        """
        kwargs = {'action': action, 'params': params, 'data': data, 'headers': headers, 'method': method}
        return kwargs

    def get_poll_request_kwargs(self, response, context, request_kwargs):
        """
        Return keyword arguments which are passed to the request() method when
        polling for the job status.

        :param response: Response object returned by poll request.
        :type response: :class:`HTTPResponse`

        :param request_kwargs: Kwargs previously used to initiate the
                                  poll request.
        :type response: ``dict``

        :return ``dict`` Keyword arguments
        """
        raise NotImplementedError('get_poll_request_kwargs not implemented')

    def has_completed(self, response):
        """
        Return job completion status.

        :param response: Response object returned by poll request.
        :type response: :class:`HTTPResponse`

        :return ``bool`` True if the job has completed, False otherwise.
        """
        raise NotImplementedError('has_completed not implemented')