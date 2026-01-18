import sys
import io
import random
import mimetypes
import time
import os
import shutil
import smtplib
import shlex
import re
import subprocess
from urllib.parse import urlencode
from urllib import parse as urlparse
from http.cookies import BaseCookie
from paste import wsgilib
from paste import lint
from paste.response import HeaderDict
class ProcResult(object):
    """
    Represents the results of running a command in
    `TestFileEnvironment
    <class-paste.fixture.TestFileEnvironment.html>`_.

    Attributes to pay particular attention to:

    ``stdout``, ``stderr``:
        What is produced

    ``files_created``, ``files_deleted``, ``files_updated``:
        Dictionaries mapping filenames (relative to the ``base_dir``)
        to `FoundFile <class-paste.fixture.FoundFile.html>`_ or
        `FoundDir <class-paste.fixture.FoundDir.html>`_ objects.
    """

    def __init__(self, test_env, args, stdin, stdout, stderr, returncode, files_before, files_after):
        self.test_env = test_env
        self.args = args
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.files_before = files_before
        self.files_after = files_after
        self.files_deleted = {}
        self.files_updated = {}
        self.files_created = files_after.copy()
        for path, f in files_before.items():
            if path not in files_after:
                self.files_deleted[path] = f
                continue
            del self.files_created[path]
            if f.mtime < files_after[path].mtime:
                self.files_updated[path] = files_after[path]

    def assert_no_error(self):
        __tracebackhide__ = True
        assert self.returncode == 0, 'Script returned code: %s' % self.returncode

    def assert_no_stderr(self):
        __tracebackhide__ = True
        if self.stderr:
            print('Error output:')
            print(self.stderr)
            raise AssertionError('stderr output not expected')

    def __str__(self):
        s = ['Script result: %s' % ' '.join(self.args)]
        if self.returncode:
            s.append('  return code: %s' % self.returncode)
        if self.stderr:
            s.append('-- stderr: --------------------')
            s.append(self.stderr)
        if self.stdout:
            s.append('-- stdout: --------------------')
            s.append(self.stdout)
        for name, files, show_size in [('created', self.files_created, True), ('deleted', self.files_deleted, True), ('updated', self.files_updated, True)]:
            if files:
                s.append('-- %s: -------------------' % name)
                files = files.items()
                files.sort()
                last = ''
                for path, f in files:
                    t = '  %s' % _space_prefix(last, path, indent=4, include_sep=False)
                    last = path
                    if show_size and f.size != 'N/A':
                        t += '  (%s bytes)' % f.size
                    s.append(t)
        return '\n'.join(s)