from __future__ import print_function
import os,stat,time
import errno
import sys
def _normalized(self, p):
    """ Make a key suitable for user's eyes """
    return str(p.relative_to(self.root)).replace('\\', '/')