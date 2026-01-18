import abc
import sys
import stat as st
from _collections_abc import _check_methods
from os.path import (curdir, pardir, sep, pathsep, defpath, extsep, altsep,
from _collections_abc import MutableMapping, Mapping
def _spawnvef(mode, file, args, env, func):
    if not isinstance(args, (tuple, list)):
        raise TypeError('argv must be a tuple or a list')
    if not args or not args[0]:
        raise ValueError('argv first element cannot be empty')
    pid = fork()
    if not pid:
        try:
            if env is None:
                func(file, args)
            else:
                func(file, args, env)
        except:
            _exit(127)
    else:
        if mode == P_NOWAIT:
            return pid
        while 1:
            wpid, sts = waitpid(pid, 0)
            if WIFSTOPPED(sts):
                continue
            return waitstatus_to_exitcode(sts)