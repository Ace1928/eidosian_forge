import contextlib
import os
import re
import subprocess
import sys
import tempfile
from io import BytesIO
from .. import diff, errors, osutils
from .. import revision as _mod_revision
from .. import revisionspec, revisiontree, tests
from ..tests import EncodingAdapter, features
from ..tests.scenarios import load_tests_apply_scenarios
class DiffWasIs(diff.DiffPath):

    def diff(self, old_path, new_path, old_kind, new_kind):
        self.to_file.write(b'was: ')
        self.to_file.write(self.old_tree.get_file(old_path).read())
        self.to_file.write(b'is: ')
        self.to_file.write(self.new_tree.get_file(new_path).read())