from __future__ import annotations
from collections import OrderedDict
from typing import TYPE_CHECKING, Any
from . import util
import re
class AndSubstitutePostprocessor(Postprocessor):
    """ Restore valid entities """

    def run(self, text: str) -> str:
        text = text.replace(util.AMP_SUBSTITUTE, '&')
        return text