from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.gandi import BaseObject
from libcloud.common.types import LibcloudError, InvalidCredsError
class LinodeResponse(JsonResponse):
    """
    Linode API response

    Wraps the HTTP response returned by the Linode API.

    libcloud does not take advantage of batching, so a response will always
    reflect the above format. A few weird quirks are caught here as well.
    """
    objects = None

    def __init__(self, response, connection):
        """Instantiate a LinodeResponse from the HTTP response

        :keyword response: The raw response returned by urllib
        :return: parsed :class:`LinodeResponse`"""
        self.errors = []
        super().__init__(response, connection)
        self.invalid = LinodeException(255, 'Invalid JSON received from server')
        self.objects, self.errors = self.parse_body()
        if not self.success():
            raise self.errors[0]

    def parse_body(self):
        """Parse the body of the response into JSON objects

        If the response chokes the parser, action and data will be returned as
        None and errorarray will indicate an invalid JSON exception.

        :return: ``list`` of objects and ``list`` of errors"""
        js = super().parse_body()
        try:
            if isinstance(js, dict):
                js = [js]
            ret = []
            errs = []
            for obj in js:
                if 'DATA' not in obj or 'ERRORARRAY' not in obj or 'ACTION' not in obj:
                    ret.append(None)
                    errs.append(self.invalid)
                    continue
                ret.append(obj['DATA'])
                errs.extend((self._make_excp(e) for e in obj['ERRORARRAY']))
            return (ret, errs)
        except Exception:
            return (None, [self.invalid])

    def success(self):
        """Check the response for success

        The way we determine success is by the presence of an error in
        ERRORARRAY.  If one is there, we assume the whole request failed.

        :return: ``bool`` indicating a successful request"""
        return len(self.errors) == 0

    def _make_excp(self, error):
        """Convert an API error to a LinodeException instance

        :keyword error: JSON object containing ``ERRORCODE`` and
        ``ERRORMESSAGE``
        :type error: dict"""
        if 'ERRORCODE' not in error or 'ERRORMESSAGE' not in error:
            return None
        if error['ERRORCODE'] == 4:
            return InvalidCredsError(error['ERRORMESSAGE'])
        return LinodeException(error['ERRORCODE'], error['ERRORMESSAGE'])