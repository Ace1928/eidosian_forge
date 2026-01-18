from novaclient.tests.functional import base
class TestFlavorNovaClientV275(TestFlavorNovaClientV274):
    """Functional tests for flavors"""
    COMPUTE_API_VERSION = '2.75'
    SWAP_DEFAULT = '0'