from __future__ import annotations
import re
import contextlib
import io
import logging
import os
import signal
from subprocess import Popen, PIPE, DEVNULL
import subprocess
import threading
from textwrap import dedent
from git.compat import defenc, force_bytes, safe_decode
from git.exc import (
from git.util import (
from typing import (
from git.types import PathLike, Literal, TBD
def _safer_popen_windows(command: Union[str, Sequence[Any]], *, shell: bool=False, env: Optional[Mapping[str, str]]=None, **kwargs: Any) -> Popen:
    """Call :class:`subprocess.Popen` on Windows but don't include a CWD in the search.

    This avoids an untrusted search path condition where a file like ``git.exe`` in a
    malicious repository would be run when GitPython operates on the repository. The
    process using GitPython may have an untrusted repository's working tree as its
    current working directory. Some operations may temporarily change to that directory
    before running a subprocess. In addition, while by default GitPython does not run
    external commands with a shell, it can be made to do so, in which case the CWD of
    the subprocess, which GitPython usually sets to a repository working tree, can
    itself be searched automatically by the shell. This wrapper covers all those cases.

    :note: This currently works by setting the ``NoDefaultCurrentDirectoryInExePath``
        environment variable during subprocess creation. It also takes care of passing
        Windows-specific process creation flags, but that is unrelated to path search.

    :note: The current implementation contains a race condition on :attr:`os.environ`.
        GitPython isn't thread-safe, but a program using it on one thread should ideally
        be able to mutate :attr:`os.environ` on another, without unpredictable results.
        See comments in https://github.com/gitpython-developers/GitPython/pull/1650.
    """
    creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    if shell:
        safer_env = {} if env is None else dict(env)
        safer_env['NoDefaultCurrentDirectoryInExePath'] = '1'
    else:
        safer_env = env
    with patch_env('NoDefaultCurrentDirectoryInExePath', '1'):
        return Popen(command, shell=shell, env=safer_env, creationflags=creationflags, **kwargs)