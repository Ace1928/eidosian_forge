from unittest import mock
from oslotest import base
from oslotest import mock_fixture
from six.moves import builtins
import os
from os_win import exceptions
from os_win.utils import baseutils
class FakeWMIExc(exceptions.x_wmi):

    def __init__(self, hresult=None):
        super(FakeWMIExc, self).__init__()
        excepinfo = [None] * 5 + [hresult]
        self.com_error = mock.Mock(excepinfo=excepinfo)
        self.com_error.hresult = hresult