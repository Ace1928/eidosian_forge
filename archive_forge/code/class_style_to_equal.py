import time
import logging
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from dash.testing.errors import TestingTimeoutError
class style_to_equal:

    def __init__(self, selector, style, val):
        self.selector = selector
        self.style = style
        self.val = val

    def __call__(self, driver):
        try:
            elem = driver.find_element(By.CSS_SELECTOR, self.selector)
            val = elem.value_of_css_property(self.style)
            logger.debug('style to equal {%s} => expected %s', val, self.val)
            return val == self.val
        except WebDriverException:
            return False