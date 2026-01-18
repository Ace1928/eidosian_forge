import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def _diff_file2_revno1(self):
    return b"=== added file 'file2'\n--- file2\t1970-01-01 00:00:00 +0000\n+++ file2\t2005-11-22 00:00:00 +0000\n@@ -0,0 +1,1 @@\n+contents of level0/file2\n\n"