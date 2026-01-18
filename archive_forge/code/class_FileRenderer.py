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
class FileRenderer(BaseRenderer):

    def __init__(self, search_path: Union[str, List[str]]) -> None:
        if isinstance(search_path, str):
            search_path = [search_path]
        else:
            search_path = list(filter(None, search_path))
        loader = SphinxFileSystemLoader(search_path)
        super().__init__(loader)

    @classmethod
    def render_from_file(cls, filename: str, context: Dict[str, Any]) -> str:
        dirname = os.path.dirname(filename)
        basename = os.path.basename(filename)
        return cls(dirname).render(basename, context)