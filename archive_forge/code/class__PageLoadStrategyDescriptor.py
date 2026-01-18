import typing
from abc import ABCMeta
from abc import abstractmethod
from enum import Enum
from selenium.common.exceptions import InvalidArgumentException
from selenium.webdriver.common.proxy import Proxy
class _PageLoadStrategyDescriptor:
    """Determines the point at which a navigation command is returned:
    https://w3c.github.io/webdriver/#dfn-table-of-page-load-strategies.

    :param strategy: the strategy corresponding to a document readiness state
    """

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, cls):
        return obj._caps.get(self.name)

    def __set__(self, obj, value):
        if value in ('normal', 'eager', 'none'):
            obj.set_capability(self.name, value)
        else:
            raise ValueError('Strategy can only be one of the following: normal, eager, none')