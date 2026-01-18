from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def __write(self, s):
    """Write string s to the stream if a whole new block
           is ready to be written.
        """
    self.buf += s
    while len(self.buf) > self.bufsize:
        self.fileobj.write(self.buf[:self.bufsize])
        self.buf = self.buf[self.bufsize:]