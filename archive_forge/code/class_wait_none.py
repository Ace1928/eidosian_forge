import abc
import random
import typing
from pip._vendor.tenacity import _utils
class wait_none(wait_fixed):
    """Wait strategy that doesn't wait at all before retrying."""

    def __init__(self) -> None:
        super().__init__(0)