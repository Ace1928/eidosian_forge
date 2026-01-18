import os
import time
from platform import system
from subprocess import DEVNULL
from subprocess import STDOUT
from subprocess import Popen
from typing_extensions import deprecated
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common import utils
def add_command_line_options(self, *args):
    self.command_line = args