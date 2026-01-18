import pytest
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
class _editable_text_input:
    """
    This function is added for convenience, as it is pretty
    cumbersome to find and update the value of Editable text input.
    """

    def __init__(self, page, nth=0):
        self.page = page
        self.text_input = page.locator('input.bk-input').nth(nth)

    @property
    def value(self):
        return self.text_input.input_value()

    @value.setter
    def value(self, value):
        self.text_input.fill(str(value))
        self.text_input.press('Enter')