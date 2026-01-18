import stevedore
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
class EntryPointTests(utils.TestCase):
    """Simple test that will check that all entry points are loadable."""

    def test_all_entry_points_are_valid(self):
        errors = []

        def raise_exception_callback(manager, entrypoint, exc):
            error = "Cannot load '%(entrypoint)s' entry_point: %(error)s'" % {'entrypoint': entrypoint, 'error': exc}
            errors.append(error)
        stevedore.ExtensionManager(namespace=loading.PLUGIN_NAMESPACE, on_load_failure_callback=raise_exception_callback)
        self.assertEqual([], errors)