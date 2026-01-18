from six.moves import range
import sys
import mock
from pyu2f import errors
from pyu2f import hidtransport
from pyu2f.tests.lib import util
def FakeHidDevice(path):
    mock_hid_dev = mock.MagicMock()
    mock_hid_dev.GetInReportDataLength.return_value = 64
    mock_hid_dev.GetOutReportDataLength.return_value = 64
    mock_hid_dev.path = path
    return mock_hid_dev