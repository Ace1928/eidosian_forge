from __future__ import division
import re
import stat
from .helpers import (
class FileDeleteAllCommand(FileCommand):

    def __init__(self):
        FileCommand.__init__(self, b'filedeleteall')

    def __bytes__(self):
        return b'deleteall'