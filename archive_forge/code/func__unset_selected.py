from typing import List
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import UnexpectedTagNameException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
def _unset_selected(self, option) -> None:
    if option.is_selected():
        option.click()