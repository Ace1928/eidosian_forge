from __future__ import (absolute_import, division, print_function)
import os
import re
import traceback
from operator import itemgetter

        Read an OpenSSH config from the given file object.
        @param file_obj: a file-like object to read the config file from
        @type file_obj: file
        