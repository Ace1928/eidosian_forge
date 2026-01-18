from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import six.moves.urllib.parse
def _check_digest(digest):
    _check_element('digest', digest, _DIGEST_CHARS, 7 + 64, 7 + 64)