import traceback
from .retries import AWSRetry
class DirectConnectError(Exception):

    def __init__(self, msg, last_traceback=None, exception=None):
        self.msg = msg
        self.last_traceback = last_traceback
        self.exception = exception