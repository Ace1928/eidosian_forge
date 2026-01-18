from ... import config, tests
from .. import script
from .. import test_config as _t_config
class TestConfigActive(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        _t_config.create_configs_with_file_option(self)

    def test_active_in_locations(self):
        script.run_script(self, '            $ brz config -d tree file\n            locations\n            ')

    def test_active_in_breezy(self):
        script.run_script(self, '            $ brz config -d tree --scope breezy file\n            breezy\n            ')

    def test_active_in_branch(self):
        script.run_script(self, '            $ brz config -d tree --scope locations --remove file\n            $ brz config -d tree file\n            branch\n            ')