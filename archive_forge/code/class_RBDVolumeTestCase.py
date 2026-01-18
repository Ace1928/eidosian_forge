from unittest import mock
from os_brick import exception
from os_brick.initiator import linuxrbd
from os_brick.tests import base
from os_brick import utils
class RBDVolumeTestCase(base.TestCase):

    def test_name_attribute(self):
        mock_client = mock.Mock()
        rbd_volume = linuxrbd.RBDVolume(mock_client, 'volume')
        self.assertEqual('volume', rbd_volume.name)