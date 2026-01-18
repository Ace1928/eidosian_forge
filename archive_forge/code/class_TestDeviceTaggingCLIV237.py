from tempest.lib import exceptions
from novaclient.tests.functional import base
class TestDeviceTaggingCLIV237(TestBlockDeviceTaggingCLIError, TestNICDeviceTaggingCLIError):
    """Tests that in microversion 2.37, creating a server with either a
    tagged block device or tagged nic would fail.
    """
    COMPUTE_API_VERSION = '2.37'