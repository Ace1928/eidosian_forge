import os
import unittest
import pytest
from monty.io import (
class TestReverseReadfile:
    NUMLINES = 3000

    def test_reverse_readfile(self):
        """
        We are making sure a file containing line numbers is read in reverse
        order, i.e. the first line that is read corresponds to the last line.
        number
        """
        fname = os.path.join(test_dir, '3000_lines.txt')
        for idx, line in enumerate(reverse_readfile(fname)):
            assert int(line) == self.NUMLINES - idx

    def test_reverse_readfile_gz(self):
        """
        We are making sure a file containing line numbers is read in reverse
        order, i.e. the first line that is read corresponds to the last line.
        number
        """
        fname = os.path.join(test_dir, '3000_lines.txt.gz')
        for idx, line in enumerate(reverse_readfile(fname)):
            assert int(line) == self.NUMLINES - idx

    def test_reverse_readfile_bz2(self):
        """
        We are making sure a file containing line numbers is read in reverse
        order, i.e. the first line that is read corresponds to the last line.
        number
        """
        fname = os.path.join(test_dir, '3000_lines.txt.bz2')
        for idx, line in enumerate(reverse_readfile(fname)):
            assert int(line) == self.NUMLINES - idx

    def test_empty_file(self):
        """
        make sure an empty file does not throw an error when reverse_readline
        is called this was a problem with an earlier implementation
        """
        for idx, line in enumerate(reverse_readfile(os.path.join(test_dir, 'empty_file.txt'))):
            raise ValueError('an empty file is being read!')