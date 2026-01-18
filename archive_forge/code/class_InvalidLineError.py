from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import os
import re
class InvalidLineError(InternalParserError):
    """Error indicating that a line of the ignore file was invalid."""