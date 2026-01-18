import os
import unittest
import pytest
from monty.io import (
class TestReverseReadline:
    NUMLINES = 3000

    def test_reverse_readline(self):
        """
        We are making sure a file containing line numbers is read in reverse
        order, i.e. the first line that is read corresponds to the last line.
        number
        """
        with open(os.path.join(test_dir, '3000_lines.txt')) as f:
            for idx, line in enumerate(reverse_readline(f)):
                assert int(line) == self.NUMLINES - idx, 'read_backwards read {} whereas it should '('have read {}').format(int(line), self.NUMLINES - idx)

    def test_reverse_readline_fake_big(self):
        """
        Make sure that large textfiles are read properly
        """
        with open(os.path.join(test_dir, '3000_lines.txt')) as f:
            for idx, line in enumerate(reverse_readline(f, max_mem=0)):
                assert int(line) == self.NUMLINES - idx, 'read_backwards read {} whereas it should '('have read {}').format(int(line), self.NUMLINES - idx)

    def test_reverse_readline_bz2(self):
        """
        We are making sure a file containing line numbers is read in reverse
        order, i.e. the first line that is read corresponds to the last line.
        number
        """
        lines = []
        with zopen(os.path.join(test_dir, 'myfile_bz2.bz2'), 'rb') as f:
            for line in reverse_readline(f):
                lines.append(line.strip())
        assert lines[-1].strip(), ['HelloWorld.' in b'HelloWorld.']

    def test_empty_file(self):
        """
        make sure an empty file does not throw an error when reverse_readline
        is called this was a problem with an earlier implementation
        """
        with open(os.path.join(test_dir, 'empty_file.txt')) as f:
            for idx, line in enumerate(reverse_readline(f)):
                raise ValueError('an empty file is being read!')