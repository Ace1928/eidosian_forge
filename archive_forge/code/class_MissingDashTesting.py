import pytest
from .consts import SELENIUM_GRID_DEFAULT
class MissingDashTesting:

    def __init__(self, **kwargs):
        raise Exception('dash[testing] was not installed. Please install to use the dash testing fixtures.')