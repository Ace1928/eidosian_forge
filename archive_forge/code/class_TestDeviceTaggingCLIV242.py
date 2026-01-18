from tempest.lib import exceptions
from novaclient.tests.functional import base
class TestDeviceTaggingCLIV242(TestBlockDeviceTaggingCLI, TestNICDeviceTaggingCLI):
    """Tests that in microversion 2.42 you could once again create a server
    with a tagged block device or a tagged nic.
    """
    COMPUTE_API_VERSION = '2.42'