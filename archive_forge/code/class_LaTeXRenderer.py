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
class LaTeXRenderer(SphinxRenderer):

    def __init__(self, template_path: Optional[str]=None, latex_engine: Optional[str]=None) -> None:
        if template_path is None:
            template_path = os.path.join(package_dir, 'templates', 'latex')
        super().__init__(template_path)
        escape = partial(texescape.escape, latex_engine=latex_engine)
        self.env.filters['e'] = escape
        self.env.filters['escape'] = escape
        self.env.filters['eabbr'] = texescape.escape_abbr
        self.env.variable_start_string = '<%='
        self.env.variable_end_string = '%>'
        self.env.block_start_string = '<%'
        self.env.block_end_string = '%>'
        self.env.comment_start_string = '<#'
        self.env.comment_end_string = '#>'