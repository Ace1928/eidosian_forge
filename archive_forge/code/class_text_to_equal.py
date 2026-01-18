import time
import logging
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from dash.testing.errors import TestingTimeoutError
class text_to_equal:

    def __init__(self, selector, text, timeout):
        self.selector = selector
        self.text = text
        self.timeout = timeout

    def __call__(self, driver):
        try:
            elem = self._get_element(driver)
            logger.debug('text to equal {%s} => expected %s', elem.text, self.text)
            return str(elem.text) == self.text or str(elem.get_attribute('value')) == self.text
        except WebDriverException:
            return False

    def message(self, driver):
        try:
            element = self._get_element(driver)
            text = 'found: ' + str(element.text) or str(element.get_attribute('value'))
        except WebDriverException:
            text = f'{self.selector} not found'
        return f'text -> {self.text} not found within {self.timeout}s, {text}'

    def _get_element(self, driver):
        return driver.find_element(By.CSS_SELECTOR, self.selector)