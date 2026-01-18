import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import XmlResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeState
class AbiquoResponse(XmlResponse):
    """
    Abiquo XML Response.

    Wraps the response in XML bodies or extract the error data in
    case of error.
    """
    NODE_STATE_MAP = {'NOT_ALLOCATED': NodeState.TERMINATED, 'ALLOCATED': NodeState.PENDING, 'CONFIGURED': NodeState.PENDING, 'ON': NodeState.RUNNING, 'PAUSED': NodeState.PENDING, 'OFF': NodeState.PENDING, 'LOCKED': NodeState.PENDING, 'UNKNOWN': NodeState.UNKNOWN}

    def parse_error(self):
        """
        Parse the error messages.

        Response body can easily be handled by this class parent
        :class:`XmlResponse`, but there are use cases which Abiquo API
        does not respond an XML but an HTML. So we need to
        handle these special cases.
        """
        if self.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError(driver=self.connection.driver)
        elif self.status == httplib.FORBIDDEN:
            raise ForbiddenError(self.connection.driver)
        elif self.status == httplib.NOT_ACCEPTABLE:
            raise LibcloudError('Not Acceptable')
        else:
            parsebody = self.parse_body()
            if parsebody is not None and hasattr(parsebody, 'findall'):
                errors = self.parse_body().findall('error')
                raise LibcloudError(errors[0].findtext('message'))
            else:
                raise LibcloudError(self.body)

    def success(self):
        """
        Determine if the request was successful.

        Any of the 2XX HTTP response codes are accepted as successful requests

        :rtype:  ``bool``
        :return: successful request or not.
        """
        return self.status in [httplib.OK, httplib.CREATED, httplib.NO_CONTENT, httplib.ACCEPTED]

    def async_success(self):
        """
        Determinate if async request was successful.

        An async_request retrieves for a task object that can be successfully
        retrieved (self.status == OK), but the asynchronous task (the body of
        the HTTP response) which we are asking for has finished with an error.
        So this method checks if the status code is 'OK' and if the task
        has finished successfully.

        :rtype:  ``bool``
        :return: successful asynchronous request or not
        """
        if self.success():
            task = self.parse_body()
            return task.findtext('state') == 'FINISHED_SUCCESSFULLY'
        else:
            return False