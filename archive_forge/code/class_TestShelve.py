from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
class TestShelve(script.TestCaseWithTransportAndScript):

    def setUp(self):
        super().setUp()
        self.run_script("\n            $ brz init test\n            Created a standalone tree (format: 2a)\n            $ cd test\n            $ echo foo > file\n            $ brz add\n            adding file\n            $ brz commit -m 'file added'\n            2>Committing to:...test/\n            2>added file\n            2>Committed revision 1.\n            $ echo bar > file\n            ")

    def test_shelve(self):
        self.run_script('\n            $ brz shelve -m \'shelve bar\'\n            2>Shelve? ([y]es, [N]o, [f]inish, [q]uit): yes\n            <y\n            2>Selected changes:\n            2> M  file\n            2>Shelve 1 change(s)? ([y]es, [N]o, [f]inish, [q]uit): yes\n            <y\n            2>Changes shelved with id "1".\n            ', null_output_matches_anything=True)
        self.run_script('\n            $ brz shelve --list\n              1: shelve bar\n            ')

    def test_dont_shelve(self):
        self.run_script("$ brz shelve -m 'shelve bar'\n2>Shelve? ([y]es, [N]o, [f]inish, [q]uit): \n2>No changes to shelve.\n", null_output_matches_anything=True)
        self.run_script('\n            $ brz st\n            modified:\n              file\n            ')