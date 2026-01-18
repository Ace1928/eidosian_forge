import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
class BannerHelpFormatter(optparse.IndentedHelpFormatter):
    """Just a small tweak to optparse to be able to print a banner."""

    def __init__(self, banner, *argv, **argd):
        self.banner = banner
        optparse.IndentedHelpFormatter.__init__(self, *argv, **argd)

    def format_usage(self, usage):
        msg = optparse.IndentedHelpFormatter.format_usage(self, usage)
        return '%s\n%s' % (self.banner, msg)