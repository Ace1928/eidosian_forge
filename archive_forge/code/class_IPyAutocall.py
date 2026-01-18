class IPyAutocall(object):
    """ Instances of this class are always autocalled
    
    This happens regardless of 'autocall' variable state. Use this to
    develop macro-like mechanisms.
    """
    _ip = None
    rewrite = True

    def __init__(self, ip=None):
        self._ip = ip

    def set_ip(self, ip):
        """Will be used to set _ip point to current ipython instance b/f call

        Override this method if you don't want this to happen.

        """
        self._ip = ip