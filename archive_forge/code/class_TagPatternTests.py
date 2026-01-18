import datetime
import os
import re
import shutil
import tempfile
import time
import unittest
from typing import ClassVar, Dict, List, Optional, Tuple
from dulwich.contrib import release_robot
from ..repo import Repo
from ..tests.utils import make_commit, make_tag
class TagPatternTests(unittest.TestCase):
    """test tag patterns."""

    def test_tag_pattern(self):
        """Test tag patterns."""
        test_cases = {'0.3': '0.3', 'v0.3': '0.3', 'release0.3': '0.3', 'Release-0.3': '0.3', 'v0.3rc1': '0.3rc1', 'v0.3-rc1': '0.3-rc1', 'v0.3-rc.1': '0.3-rc.1', 'version 0.3': '0.3', 'version_0.3_rc_1': '0.3_rc_1', 'v1': '1', '0.3rc1': '0.3rc1'}
        for testcase, version in test_cases.items():
            matches = re.match(release_robot.PATTERN, testcase)
            self.assertEqual(matches.group(1), version)