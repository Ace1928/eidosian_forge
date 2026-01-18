import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
class TestPushOutput(script.TestCaseWithTransportAndScript):

    def test_push_log_format(self):
        self.run_script("\n            $ brz init trunk\n            Created a standalone tree (format: 2a)\n            $ cd trunk\n            $ echo foo > file\n            $ brz add\n            adding file\n            $ brz commit -m 'we need some foo'\n            2>Committing to:...trunk/\n            2>added file\n            2>Committed revision 1.\n            $ brz init ../feature\n            Created a standalone tree (format: 2a)\n            $ brz push -v ../feature -Olog_format=line\n            Added Revisions:\n            1: jrandom@example.com ...we need some foo\n            2>All changes applied successfully.\n            2>Pushed up to revision 1.\n            ")

    def test_push_with_revspec(self):
        self.run_script("\n            $ brz init-shared-repo .\n            Shared repository with trees (format: 2a)\n            Location:\n              shared repository: .\n            $ brz init trunk\n            Created a repository tree (format: 2a)\n            Using shared repository...\n            $ cd trunk\n            $ brz commit -m 'first rev' --unchanged\n            2>Committing to:...trunk/\n            2>Committed revision 1.\n            $ echo foo > file\n            $ brz add\n            adding file\n            $ brz commit -m 'we need some foo'\n            2>Committing to:...trunk/\n            2>added file\n            2>Committed revision 2.\n            $ brz push -r 1 ../other\n            2>Created new branch.\n            $ brz st ../other # checking that file is not created (#484516)\n            ")