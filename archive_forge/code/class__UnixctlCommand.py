import sys
import ovs.util
class _UnixctlCommand(object):

    def __init__(self, usage, min_args, max_args, callback, aux):
        self.usage = usage
        self.min_args = min_args
        self.max_args = max_args
        self.callback = callback
        self.aux = aux