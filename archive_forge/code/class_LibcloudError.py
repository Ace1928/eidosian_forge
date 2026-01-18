from enum import Enum
from typing import Union, Callable, Optional, cast
class LibcloudError(Exception):
    """The base class for other libcloud exceptions"""

    def __init__(self, value, driver=None):
        super().__init__(value)
        self.value = value
        self.driver = driver

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<LibcloudError in ' + repr(self.driver) + ' ' + repr(self.value) + '>'