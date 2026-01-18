from io import StringIO
import breezy.commands
from . import TestCase
class TestGenerateDocs(TestCase):

    def setUp(self):
        super().setUp()
        self.sio = StringIO()
        self.options = Options()
        self.options.brz_name = 'brz'
        breezy.commands.install_bzr_command_hooks()

    def test_man_page(self):
        from breezy.doc_generate import autodoc_man
        autodoc_man.infogen(self.options, self.sio)
        self.assertNotEqual('', self.sio.getvalue())

    def test_rstx_man(self):
        from breezy.doc_generate import autodoc_rstx
        autodoc_rstx.infogen(self.options, self.sio)
        self.assertNotEqual('', self.sio.getvalue())