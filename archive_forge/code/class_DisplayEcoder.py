from __future__ import annotations
import sys
from io import StringIO
from json import JSONEncoder, loads
from typing import TYPE_CHECKING
class DisplayEcoder(JSONEncoder):
    """
    Help convert dicts and objects to a format that can be displayed in notebooks
    """

    def default(self, o):
        """
        Try different ways of converting the present object for displaying
        """
        try:
            return o.as_dict()
        except Exception:
            pass
        try:
            return o.__dict__
        except Exception:
            pass
        try:
            return str(o)
        except Exception:
            pass
        return None