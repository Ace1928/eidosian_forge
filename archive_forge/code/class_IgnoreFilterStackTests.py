import os
import re
import shutil
import tempfile
from io import BytesIO
from dulwich.tests import TestCase
from ..ignore import (
from ..repo import Repo
class IgnoreFilterStackTests(TestCase):

    def test_stack_first(self):
        filter1 = IgnoreFilter([b'[a].c', b'[b].c', b'![d].c'])
        filter2 = IgnoreFilter([b'[a].c', b'![b],c', b'[c].c', b'[d].c'])
        stack = IgnoreFilterStack([filter1, filter2])
        self.assertIs(True, stack.is_ignored(b'a.c'))
        self.assertIs(True, stack.is_ignored(b'b.c'))
        self.assertIs(True, stack.is_ignored(b'c.c'))
        self.assertIs(False, stack.is_ignored(b'd.c'))
        self.assertIs(None, stack.is_ignored(b'e.c'))