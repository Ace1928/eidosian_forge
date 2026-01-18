class U2FError(Exception):
    OK = 0
    OTHER_ERROR = 1
    BAD_REQUEST = 2
    CONFIGURATION_UNSUPPORTED = 3
    DEVICE_INELIGIBLE = 4
    TIMEOUT = 5

    def __init__(self, code, cause=None):
        self.code = code
        if cause:
            self.cause = cause
        super(U2FError, self).__init__('U2F Error code: %d (cause: %s)' % (code, str(cause)))