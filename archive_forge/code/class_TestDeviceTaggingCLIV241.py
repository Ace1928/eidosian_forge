from tempest.lib import exceptions
from novaclient.tests.functional import base
class TestDeviceTaggingCLIV241(TestBlockDeviceTaggingCLIError, TestNICDeviceTaggingCLIError):
    """Tests that in microversion 2.41, creating a server with either a
    tagged block device or tagged nic would fail. This is testing the
    boundary before 2.42 where block device tags and nic tags were fixed
    for server create requests.
    """
    COMPUTE_API_VERSION = '2.41'