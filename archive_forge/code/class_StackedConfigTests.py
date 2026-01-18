import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
class StackedConfigTests(TestCase):

    def test_default_backends(self):
        StackedConfig.default_backends()

    @skipIf(sys.platform != 'win32', 'Windows specific config location.')
    def test_windows_config_from_path(self):
        from ..config import get_win_system_paths
        install_dir = os.path.join('C:', 'foo', 'Git')
        self.overrideEnv('PATH', os.path.join(install_dir, 'cmd'))
        with patch('os.path.exists', return_value=True):
            paths = set(get_win_system_paths())
        self.assertEqual({os.path.join(os.environ.get('PROGRAMDATA'), 'Git', 'config'), os.path.join(install_dir, 'etc', 'gitconfig')}, paths)

    @skipIf(sys.platform != 'win32', 'Windows specific config location.')
    def test_windows_config_from_reg(self):
        import winreg
        from ..config import get_win_system_paths
        self.overrideEnv('PATH', None)
        install_dir = os.path.join('C:', 'foo', 'Git')
        with patch('winreg.OpenKey'):
            with patch('winreg.QueryValueEx', return_value=(install_dir, winreg.REG_SZ)):
                paths = set(get_win_system_paths())
        self.assertEqual({os.path.join(os.environ.get('PROGRAMDATA'), 'Git', 'config'), os.path.join(install_dir, 'etc', 'gitconfig')}, paths)