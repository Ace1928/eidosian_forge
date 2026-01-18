import typing
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from .abstract_event_listener import AbstractEventListener
def _wrap_elements(result, ef_driver):
    if isinstance(result, EventFiringWebElement):
        return result
    if isinstance(result, WebElement):
        return EventFiringWebElement(result, ef_driver)
    if isinstance(result, list):
        return [_wrap_elements(item, ef_driver) for item in result]
    return result