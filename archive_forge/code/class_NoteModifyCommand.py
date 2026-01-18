from __future__ import division
import re
import stat
from .helpers import (
class NoteModifyCommand(FileCommand):

    def __init__(self, from_, data):
        super(NoteModifyCommand, self).__init__(b'notemodify')
        self.from_ = from_
        self.data = data
        self._binary = ['data']

    def __bytes__(self):
        return b'N inline :' + self.from_ + ('\ndata %d\n' % len(self.data)).encode('ascii') + self.data