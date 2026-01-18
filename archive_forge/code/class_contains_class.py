import time
import logging
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from dash.testing.errors import TestingTimeoutError
class contains_class:

    def __init__(self, selector, classname):
        self.selector = selector
        self.classname = classname

    def __call__(self, driver):
        try:
            elem = driver.find_element(By.CSS_SELECTOR, self.selector)
            classname = elem.get_attribute('class')
            logger.debug('contains class {%s} => expected %s', classname, self.classname)
            return self.classname in str(classname).split(' ')
        except WebDriverException:
            return False