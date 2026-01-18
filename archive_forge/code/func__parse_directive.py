import fnmatch
import logging
import os
import re
import sys
from . import DistlibException
from .compat import fsdecode
from .util import convert_path
def _parse_directive(self, directive):
    """
        Validate a directive.
        :param directive: The directive to validate.
        :return: A tuple of action, patterns, thedir, dir_patterns
        """
    words = directive.split()
    if len(words) == 1 and words[0] not in ('include', 'exclude', 'global-include', 'global-exclude', 'recursive-include', 'recursive-exclude', 'graft', 'prune'):
        words.insert(0, 'include')
    action = words[0]
    patterns = thedir = dir_pattern = None
    if action in ('include', 'exclude', 'global-include', 'global-exclude'):
        if len(words) < 2:
            raise DistlibException('%r expects <pattern1> <pattern2> ...' % action)
        patterns = [convert_path(word) for word in words[1:]]
    elif action in ('recursive-include', 'recursive-exclude'):
        if len(words) < 3:
            raise DistlibException('%r expects <dir> <pattern1> <pattern2> ...' % action)
        thedir = convert_path(words[1])
        patterns = [convert_path(word) for word in words[2:]]
    elif action in ('graft', 'prune'):
        if len(words) != 2:
            raise DistlibException('%r expects a single <dir_pattern>' % action)
        dir_pattern = convert_path(words[1])
    else:
        raise DistlibException('unknown action %r' % action)
    return (action, patterns, thedir, dir_pattern)