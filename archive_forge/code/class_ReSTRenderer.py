import os
from functools import partial
from os import path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from jinja2 import TemplateNotFound
from jinja2.environment import Environment
from jinja2.loaders import BaseLoader
from jinja2.sandbox import SandboxedEnvironment
from sphinx import package_dir
from sphinx.jinja2glue import SphinxFileSystemLoader
from sphinx.locale import get_translator
from sphinx.util import rst, texescape
class ReSTRenderer(SphinxRenderer):

    def __init__(self, template_path: Union[None, str, List[str]]=None, language: Optional[str]=None) -> None:
        super().__init__(template_path)
        self.env.extend(language=language)
        self.env.filters['e'] = rst.escape
        self.env.filters['escape'] = rst.escape
        self.env.filters['heading'] = rst.heading