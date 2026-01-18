from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import six.moves.urllib.parse
def _check_repository(repository):
    _check_element('repository', repository, _REPOSITORY_CHARS, 2, 255)