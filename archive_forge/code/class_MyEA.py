import numpy as np
import pytest
from pandas.core.arrays import ExtensionArray
class MyEA(ExtensionArray):

    def __init__(self, values) -> None:
        self._values = values