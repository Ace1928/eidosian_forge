import os
import time
from platform import system
from subprocess import DEVNULL
from subprocess import STDOUT
from subprocess import Popen
from typing_extensions import deprecated
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common import utils
def _start_from_profile_path(self, path):
    self._firefox_env['XRE_PROFILE_PATH'] = path
    if self.platform == 'linux':
        self._modify_link_library_path()
    command = [self._start_cmd, '-foreground']
    if self.command_line:
        for cli in self.command_line:
            command.append(cli)
    self.process = Popen(command, stdout=self._log_file, stderr=STDOUT, env=self._firefox_env)