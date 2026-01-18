from Xlib import X
from Xlib.protocol import rq
class CatchError:

    def __init__(self, *errors):
        self.error_types = errors
        self.error = None
        self.request = None

    def __call__(self, error, request):
        if self.error_types:
            for etype in self.error_types:
                if isinstance(error, etype):
                    self.error = error
                    self.request = request
                    return 1
            return 0
        else:
            self.error = error
            self.request = request
            return 1

    def get_error(self):
        return self.error

    def get_request(self):
        return self.request

    def reset(self):
        self.error = None
        self.request = None