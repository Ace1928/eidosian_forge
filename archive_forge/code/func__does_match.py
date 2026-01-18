import fnmatch
import getpass
import os
import re
import shlex
import socket
from hashlib import sha1
from io import StringIO
from functools import partial
from .ssh_exception import CouldNotCanonicalize, ConfigParseError
def _does_match(self, match_list, target_hostname, canonical, final, options):
    matched = []
    candidates = match_list[:]
    local_username = getpass.getuser()
    while candidates:
        candidate = candidates.pop(0)
        passed = None
        configured_host = options.get('hostname', None)
        configured_user = options.get('user', None)
        type_, param = (candidate['type'], candidate['param'])
        if type_ == 'canonical':
            if self._should_fail(canonical, candidate):
                return False
        if type_ == 'final':
            passed = final
        elif type_ == 'all':
            return True
        elif type_ == 'host':
            hostval = configured_host or target_hostname
            passed = self._pattern_matches(param, hostval)
        elif type_ == 'originalhost':
            passed = self._pattern_matches(param, target_hostname)
        elif type_ == 'user':
            user = configured_user or local_username
            passed = self._pattern_matches(param, user)
        elif type_ == 'localuser':
            passed = self._pattern_matches(param, local_username)
        elif type_ == 'exec':
            exec_cmd = self._tokenize(options, target_hostname, 'match-exec', param)
            if invoke is None:
                raise invoke_import_error
            passed = invoke.run(exec_cmd, hide='stdout', warn=True).ok
        if passed is not None and self._should_fail(passed, candidate):
            return False
        matched.append(candidate)
    return matched