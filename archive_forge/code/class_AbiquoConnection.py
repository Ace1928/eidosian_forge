import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import XmlResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeState
class AbiquoConnection(ConnectionUserAndKey, PollingConnection):
    """
    A Connection to Abiquo API.

    Basic :class:`ConnectionUserAndKey` connection with
    :class:`PollingConnection` features for asynchronous tasks.
    """
    responseCls = AbiquoResponse

    def __init__(self, user_id, key, secure=True, host=None, port=None, url=None, timeout=None, retry_delay=None, backoff=None, proxy_url=None):
        super().__init__(user_id=user_id, key=key, secure=secure, host=host, port=port, url=url, timeout=timeout, retry_delay=retry_delay, backoff=backoff, proxy_url=proxy_url)
        self.cache = {}

    def add_default_headers(self, headers):
        """
        Add Basic Authentication header to all the requests.

        It injects the 'Authorization: Basic Base64String===' header
        in each request

        :type  headers: ``dict``
        :param headers: Default input headers

        :rtype:         ``dict``
        :return:        Default input headers with the 'Authorization'
                        header
        """
        b64string = b('{}:{}'.format(self.user_id, self.key))
        encoded = base64.b64encode(b64string).decode('utf-8')
        authorization = 'Basic ' + encoded
        headers['Authorization'] = authorization
        return headers

    def get_poll_request_kwargs(self, response, context, request_kwargs):
        """
        Manage polling request arguments.

        Return keyword arguments which are passed to the
        :class:`NodeDriver.request` method when polling for the job status. The
        Abiquo Asynchronous Response returns and 'acceptedrequest' XmlElement
        as the following::

            <acceptedrequest>
                <link href="http://uri/to/task" rel="status"/>
                <message>You can follow the progress in the link</message>
            </acceptedrequest>

        We need to extract the href URI to poll.

        :type    response:       :class:`xml.etree.ElementTree`
        :keyword response:       Object returned by poll request.
        :type    request_kwargs: ``dict``
        :keyword request_kwargs: Default request arguments and headers
        :rtype:                  ``dict``
        :return:                 Modified keyword arguments
        """
        accepted_request_obj = response.object
        link_poll = get_href(accepted_request_obj, 'status')
        hdr_poll = {'Accept': 'application/vnd.abiquo.task+xml'}
        request_kwargs['action'] = link_poll
        request_kwargs['method'] = 'GET'
        request_kwargs['headers'] = hdr_poll
        return request_kwargs

    def has_completed(self, response):
        """
        Decide if the asynchronous job has ended.

        :type  response: :class:`xml.etree.ElementTree`
        :param response: Response object returned by poll request
        :rtype:          ``bool``
        :return:         Whether the job has completed
        """
        task = response.object
        task_state = task.findtext('state')
        return task_state in ['FINISHED_SUCCESSFULLY', 'ABORTED', 'FINISHED_UNSUCCESSFULLY']