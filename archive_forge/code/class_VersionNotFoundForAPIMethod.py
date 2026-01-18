import inspect
import sys
from zunclient.i18n import _
class VersionNotFoundForAPIMethod(Exception):
    msg_fmt = "API version '%(vers)s' is not supported on '%(method)s' method."

    def __init__(self, version, method):
        self.version = version
        self.method = method

    def __str__(self):
        return self.msg_fmt % {'vers': self.version, 'method': self.method}