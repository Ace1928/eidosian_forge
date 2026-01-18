import os
import sys
from breezy import (branch, debug, osutils, tests, uncommit, urlutils,
from breezy.bzr import remote
from breezy.directory_service import directories
from breezy.tests import fixtures, script
class TestPullOutput(script.TestCaseWithTransportAndScript):

    def test_pull_log_format(self):
        self.run_script("\n            $ brz init trunk\n            Created a standalone tree (format: 2a)\n            $ cd trunk\n            $ echo foo > file\n            $ brz add\n            adding file\n            $ brz commit -m 'we need some foo'\n            2>Committing to:...trunk/\n            2>added file\n            2>Committed revision 1.\n            $ cd ..\n            $ brz init feature\n            Created a standalone tree (format: 2a)\n            $ cd feature\n            $ brz pull -v ../trunk -Olog_format=line\n            Now on revision 1.\n            Added Revisions:\n            1: jrandom@example.com ...we need some foo\n            2>+N  file\n            2>All changes applied successfully.\n            ")