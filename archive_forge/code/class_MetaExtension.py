from __future__ import annotations
from . import Extension
from ..preprocessors import Preprocessor
import re
import logging
from typing import Any
class MetaExtension(Extension):
    """ Meta-Data extension for Python-Markdown. """

    def extendMarkdown(self, md):
        """ Add `MetaPreprocessor` to Markdown instance. """
        md.registerExtension(self)
        self.md = md
        md.preprocessors.register(MetaPreprocessor(md), 'meta', 27)

    def reset(self) -> None:
        self.md.Meta = {}