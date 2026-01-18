from enum import Enum
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.options import ArgOptions
def add_additional_option(self, name: str, value):
    """Adds an additional option not yet added as a safe option for IE.

        :Args:
         - name: name of the option to add
         - value: value of the option to add
        """
    self._additional[name] = value