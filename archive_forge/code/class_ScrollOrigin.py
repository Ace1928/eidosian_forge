from typing import Union
from selenium.webdriver.remote.webelement import WebElement
from . import interaction
from .input_device import InputDevice
class ScrollOrigin:

    def __init__(self, origin: Union[str, WebElement], x_offset: int, y_offset: int) -> None:
        self._origin = origin
        self._x_offset = x_offset
        self._y_offset = y_offset

    @classmethod
    def from_element(cls, element: WebElement, x_offset: int=0, y_offset: int=0):
        return cls(element, x_offset, y_offset)

    @classmethod
    def from_viewport(cls, x_offset: int=0, y_offset: int=0):
        return cls('viewport', x_offset, y_offset)

    @property
    def origin(self) -> Union[str, WebElement]:
        return self._origin

    @property
    def x_offset(self) -> int:
        return self._x_offset

    @property
    def y_offset(self) -> int:
        return self._y_offset