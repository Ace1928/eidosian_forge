from __future__ import annotations
import logging # isort:skip
import webbrowser
from os.path import abspath
from typing import Literal, Protocol, cast
from ..settings import settings
class DummyWebBrowser:
    """ A "no-op" web-browser controller.

    """

    def open(self, url: str, new: TargetCode=0, autoraise: bool=True) -> bool:
        """ Receive standard arguments and take no action. """
        return True