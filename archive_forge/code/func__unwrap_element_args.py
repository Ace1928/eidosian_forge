import typing
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from .abstract_event_listener import AbstractEventListener
def _unwrap_element_args(self, args):
    if isinstance(args, EventFiringWebElement):
        return args.wrapped_element
    if isinstance(args, tuple):
        return tuple((self._unwrap_element_args(item) for item in args))
    if isinstance(args, list):
        return [self._unwrap_element_args(item) for item in args]
    return args