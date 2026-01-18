from unittest import mock
from oslotest import base
from oslotest import mock_fixture
from six.moves import builtins
import os
from os_win import exceptions
from os_win.utils import baseutils
def _patch_autospec_classes(self):
    for class_type in self._autospec_classes:
        mocked_class = mock.MagicMock(autospec=class_type)
        patcher = mock.patch('.'.join([class_type.__module__, class_type.__name__]), mocked_class)
        patcher.start()