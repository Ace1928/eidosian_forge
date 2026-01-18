from __future__ import annotations
import gettext
import importlib
import json
import locale
import os
import re
import sys
import traceback
from functools import lru_cache
from typing import Any, Pattern
import babel
from packaging.version import parse as parse_version
def _np(self, msgctxt: str, msgid: str, msgid_plural: str, n: int) -> str:
    """
        Shorthand for npgettext.

        Parameters
        ----------
        msgctxt: str
            The message context.
        msgid: str
            The singular string to translate.
        msgid_plural: str
            The plural string to translate.
        n: int
            The number for pluralization.

        Returns
        -------
        str
            The translated string.
        """
    return self.npgettext(msgctxt, msgid, msgid_plural, n)