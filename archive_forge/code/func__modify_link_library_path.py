import os
import time
from platform import system
from subprocess import DEVNULL
from subprocess import STDOUT
from subprocess import Popen
from typing_extensions import deprecated
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common import utils
def _modify_link_library_path(self):
    existing_ld_lib_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_lib_path = self._extract_and_check(self.profile, 'x86', 'amd64')
    new_ld_lib_path += existing_ld_lib_path
    self._firefox_env['LD_LIBRARY_PATH'] = new_ld_lib_path
    self._firefox_env['LD_PRELOAD'] = self.NO_FOCUS_LIBRARY_NAME