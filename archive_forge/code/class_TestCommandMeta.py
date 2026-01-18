import logging
import testtools
from neutronclient.neutron import v2_0 as neutronV20
class TestCommandMeta(testtools.TestCase):

    def test_neutron_command_meta_defines_log(self):

        class FakeCommand(neutronV20.NeutronCommand):
            pass
        self.assertTrue(hasattr(FakeCommand, 'log'))
        self.assertIsInstance(FakeCommand.log, logging.getLoggerClass())
        self.assertEqual(__name__ + '.FakeCommand', FakeCommand.log.name)

    def test_neutron_command_log_defined_explicitly(self):

        class FakeCommand(neutronV20.NeutronCommand):
            log = None
        self.assertTrue(hasattr(FakeCommand, 'log'))
        self.assertIsNone(FakeCommand.log)