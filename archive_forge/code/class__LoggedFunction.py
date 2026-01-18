import traceback, logging
from OpenGL._configflags import ERROR_LOGGING, FULL_LOGGING
class _LoggedFunction(object):
    """Proxy that overrides __call__ to log arguments"""

    def __init__(self, base, log):
        self.__dict__[''] = base
        self.__dict__['log'] = log

    def __setattr__(self, key, value):
        if key != '':
            setattr(self.__dict__[''], key, value)
        else:
            self.__dict__[''] = value

    def __getattr__(self, key):
        if key == '':
            return self.__dict__['']
        else:
            return getattr(self.__dict__[''], key)