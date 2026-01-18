import inspect
import json
import sys
import traceback
class ResponseInfo(Information):
    """Response information"""

    def _func(self, conv=None):
        self._status = self.status
        _msg = conv.last_content
        if isinstance(_msg, str):
            self._message = _msg
        else:
            self._message = _msg.to_dict()
        return {}