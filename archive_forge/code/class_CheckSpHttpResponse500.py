import inspect
import json
import sys
import traceback
class CheckSpHttpResponse500(Error):
    """Checks that the SP's HTTP response status is >= 500. This is useful
    to check if the SP correctly flags errors such as an invalid signature
    """
    cid = 'check-sp-http-response-500'
    msg = 'SP does not return a HTTP 5xx status when it shold do so.'

    def _func(self, conv):
        _response = conv.last_response
        res = {}
        if _response.status_code < 500:
            self._status = self.status
            self._message = self.msg
            res['url'] = conv.position
            res['http_status'] = _response.status_code
        return res