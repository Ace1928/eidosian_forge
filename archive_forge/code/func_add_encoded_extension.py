import base64
import os
from typing import BinaryIO
from typing import List
from typing import Union
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.options import ArgOptions
def add_encoded_extension(self, extension: str) -> None:
    """Adds Base64 encoded string with extension data to a list that will
        be used to extract it to the ChromeDriver.

        :Args:
         - extension: Base64 encoded string with extension data
        """
    if extension:
        self._extensions.append(extension)
    else:
        raise ValueError('argument can not be null')