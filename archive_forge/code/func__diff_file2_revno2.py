import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def _diff_file2_revno2(self):
    return b"=== modified file 'file2'\n--- file2\t2005-11-22 00:00:00 +0000\n+++ file2\t2005-11-22 00:00:01 +0000\n@@ -1,1 +1,1 @@\n-contents of level0/file2\n+hello\n\n"