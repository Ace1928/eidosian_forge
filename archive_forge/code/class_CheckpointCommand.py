from __future__ import division
import re
import stat
from .helpers import (
class CheckpointCommand(ImportCommand):

    def __init__(self):
        ImportCommand.__init__(self, b'checkpoint')

    def __bytes__(self):
        return b'checkpoint'