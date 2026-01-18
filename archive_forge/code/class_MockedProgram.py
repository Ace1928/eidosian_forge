import functools
import logging
import os
import pipes
import shutil
import sys
import tempfile
import time
import unittest
from humanfriendly.compat import StringIO
from humanfriendly.text import random_string
class MockedProgram(CustomSearchPath):
    """
    Context manager to mock the existence of a program (executable).

    This class extends the functionality of :class:`CustomSearchPath`.
    """

    def __init__(self, name, returncode=0, script=None):
        """
        Initialize a :class:`MockedProgram` object.

        :param name: The name of the program (a string).
        :param returncode: The return code that the program should emit (a
                           number, defaults to zero).
        :param script: Shell script code to include in the mocked program (a
                       string or :data:`None`). This can be used to mock a
                       program that is expected to generate specific output.
        """
        self.program_name = name
        self.program_returncode = returncode
        self.program_script = script
        self.program_signal_file = None
        super(MockedProgram, self).__init__()

    def __enter__(self):
        """
        Create the mock program.

        :returns: The pathname of the directory that has
                  been added to ``$PATH`` (a string).
        """
        directory = super(MockedProgram, self).__enter__()
        self.program_signal_file = os.path.join(directory, 'program-was-run-%s' % random_string(10))
        pathname = os.path.join(directory, self.program_name)
        with open(pathname, 'w') as handle:
            handle.write('#!/bin/sh\n')
            handle.write('echo > %s\n' % pipes.quote(self.program_signal_file))
            if self.program_script:
                handle.write('%s\n' % self.program_script.strip())
            handle.write('exit %i\n' % self.program_returncode)
        os.chmod(pathname, 493)
        return directory

    def __exit__(self, *args, **kw):
        """
        Ensure that the mock program was run.

        :raises: :exc:`~exceptions.AssertionError` when
                 the mock program hasn't been run.
        """
        try:
            assert self.program_signal_file and os.path.isfile(self.program_signal_file), 'It looks like %r was never run!' % self.program_name
        finally:
            return super(MockedProgram, self).__exit__(*args, **kw)