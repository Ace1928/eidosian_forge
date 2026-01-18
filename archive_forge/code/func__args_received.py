from collections import deque
from io import BytesIO
from ... import debug, errors
from ...trace import mutter
def _args_received(self, args):
    self.expecting = 'body'
    self.request_handler.args_received(args)
    if self.request_handler.finished_reading:
        self._response_sent = True
        self.responder.send_response(self.request_handler.response)
        self.expecting = 'end'