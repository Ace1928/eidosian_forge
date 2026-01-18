from ironicclient.tests.functional.osc.v1 import base
class BaremetalConductorTests(base.TestCase):
    """Functional tests for baremetal conductor commands."""

    def test_list(self):
        """List available conductors.

        There is at lease one conductor in the functional tests, if not, other
        tests will fail too.
        """
        hostnames = [c['Hostname'] for c in self.conductor_list()]
        self.assertIsNotNone(hostnames)

    def test_show(self):
        """Show specified conductor.

        Conductor name varies in different environment, list first, then show
        one of them.
        """
        conductors = self.conductor_list()
        conductor = self.conductor_show(conductors[0]['Hostname'])
        self.assertIn('conductor_group', conductor)
        self.assertIn('alive', conductor)
        self.assertIn('drivers', conductor)