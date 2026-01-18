import inspect
import json
import sys
import traceback
class CheckSpHttpResponseOK(Error):
    """
    Checks that the SP's HTTP response status is within the 200 or 300 range
    """
    cid = 'check-sp-http-response-ok'
    msg = 'SP error OK'

    def _func(self, conv):
        _response = conv.last_response
        res = {}
        if _response.status_code >= 400:
            self._status = self.status
            self._message = self.msg
            res['url'] = conv.position
            res['http_status'] = _response.status_code
        return res