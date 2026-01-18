from typing import Optional
from typing import Union
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoSuchFrameException
from selenium.common.exceptions import NoSuchWindowException
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from .command import Command
@property
def active_element(self) -> WebElement:
    """Returns the element with focus, or BODY if nothing has focus.

        :Usage:
            ::

                element = driver.switch_to.active_element
        """
    return self._driver.execute(Command.W3C_GET_ACTIVE_ELEMENT)['value']