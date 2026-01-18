from io import BytesIO
import breezy.bzr.bzrdir
import breezy.mergeable
import breezy.transport
import breezy.urlutils
from ... import errors, tests
from ...tests.per_transport import transport_test_permutations
from ...tests.scenarios import load_tests_apply_scenarios
from ...tests.test_transport import TestTransportImplementation
from ..bundle.serializer import write_bundle
class TestReadMergeableBundleFromURL(TestTransportImplementation):
    """Test that read_bundle works properly across multiple transports"""
    scenarios = transport_test_permutations()

    def setUp(self):
        super().setUp()
        self.bundle_name = 'test_bundle'
        self.possible_transports = [self.get_transport(self.bundle_name)]
        self.overrideEnv('BRZ_NO_SMART_VFS', None)
        self.create_test_bundle()

    def read_mergeable_from_url(self, url):
        return breezy.mergeable.read_mergeable_from_url(url, possible_transports=self.possible_transports)

    def get_url(self, relpath=''):
        return breezy.urlutils.join(self._server.get_url(), relpath)

    def create_test_bundle(self):
        out, wt = create_bundle_file(self)
        if self.get_transport().is_readonly():
            self.build_tree_contents([(self.bundle_name, out.getvalue())])
        else:
            self.get_transport().put_file(self.bundle_name, out)
            self.log('Put to: %s', self.get_url(self.bundle_name))
        return wt

    def test_read_mergeable_from_url(self):
        info = self.read_mergeable_from_url(str(self.get_url(self.bundle_name)))
        revision = info.real_revisions[-1]
        self.assertEqual(b'commit-1', revision.revision_id)

    def test_read_fail(self):
        self.assertRaises(errors.NotABundle, self.read_mergeable_from_url, self.get_url('tree'))
        self.assertRaises(errors.NotABundle, self.read_mergeable_from_url, self.get_url('tree/a'))

    def test_read_mergeable_respects_possible_transports(self):
        if not isinstance(self.get_transport(self.bundle_name), breezy.transport.ConnectedTransport):
            raise tests.TestSkipped('Need a ConnectedTransport to test transport reuse')
        url = str(self.get_url(self.bundle_name))
        self.read_mergeable_from_url(url)
        self.assertEqual(1, len(self.possible_transports))