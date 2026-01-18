import os
import time
from platform import system
from subprocess import DEVNULL
from subprocess import STDOUT
from subprocess import Popen
from typing_extensions import deprecated
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common import utils
def _extract_and_check(self, profile, x86, amd64):
    paths = [x86, amd64]
    built_path = ''
    for path in paths:
        library_path = os.path.join(profile.path, path)
        if not os.path.exists(library_path):
            os.makedirs(library_path)
        import shutil
        shutil.copy(os.path.join(os.path.dirname(__file__), path, self.NO_FOCUS_LIBRARY_NAME), library_path)
        built_path += library_path + ':'
    return built_path