import configparser
from os import path
from typing import Dict, Optional
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.errors import ThemeError
from sphinx.locale import __
from sphinx.util import logging
class BuiltInTheme(Theme):
    """A built-in LaTeX theme."""

    def __init__(self, name: str, config: Config) -> None:
        super().__init__(name)
        if name == 'howto':
            self.docclass = config.latex_docclass.get('howto', 'article')
        else:
            self.docclass = config.latex_docclass.get('manual', 'report')
        if name in ('manual', 'howto'):
            self.wrapperclass = 'sphinx' + name
        else:
            self.wrapperclass = name
        if name == 'howto' and (not self.docclass.startswith('j')):
            self.toplevel_sectioning = 'section'
        else:
            self.toplevel_sectioning = 'chapter'