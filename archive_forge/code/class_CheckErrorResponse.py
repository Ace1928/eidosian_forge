import inspect
import json
import sys
import traceback
class CheckErrorResponse(ExpectedError):
    """
    Checks that the HTTP response status is outside the 200 or 300 range
    or that an JSON encoded error message has been received
    """
    cid = 'check-error-response'
    msg = 'OP error'

    def _func(self, conv):
        _response = conv.last_response
        _content = conv.last_content
        res = {}
        if _response.status_code >= 400:
            content_type = _response.headers['content-type']
            if content_type is None:
                res['content'] = _content
            else:
                res['content'] = _content
        return res