from ... import config, tests
from .. import script
from .. import test_config as _t_config
class TestConfigSetOption(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        _t_config.create_configs(self)

    def test_unknown_config(self):
        self.run_bzr_error(['The "moon" configuration does not exist'], ['config', '--scope', 'moon', 'hello=world'])

    def test_breezy_config_outside_branch(self):
        script.run_script(self, '            $ brz config --scope breezy hello=world\n            $ brz config -d tree --all hello\n            breezy:\n              [DEFAULT]\n              hello = world\n            ')

    def test_breezy_config_inside_branch(self):
        script.run_script(self, '            $ brz config -d tree --scope breezy hello=world\n            $ brz config -d tree --all hello\n            breezy:\n              [DEFAULT]\n              hello = world\n            ')

    def test_locations_config_inside_branch(self):
        script.run_script(self, '            $ brz config -d tree --scope locations hello=world\n            $ brz config -d tree --all hello\n            locations:\n              [.../work/tree]\n              hello = world\n            ')

    def test_branch_config_default(self):
        script.run_script(self, '            $ brz config -d tree hello=world\n            $ brz config -d tree --all hello\n            branch:\n              hello = world\n            ')

    def test_branch_config_forcing_branch(self):
        script.run_script(self, '            $ brz config -d tree --scope branch hello=world\n            $ brz config -d tree --all hello\n            branch:\n              hello = world\n            ')