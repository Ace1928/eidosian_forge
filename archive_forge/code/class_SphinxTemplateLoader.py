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
class SphinxTemplateLoader(BaseLoader):
    """A loader supporting template inheritance"""

    def __init__(self, confdir: str, templates_paths: List[str], system_templates_paths: List[str]) -> None:
        self.loaders = []
        self.sysloaders = []
        for templates_path in templates_paths:
            loader = SphinxFileSystemLoader(path.join(confdir, templates_path))
            self.loaders.append(loader)
        for templates_path in system_templates_paths:
            loader = SphinxFileSystemLoader(templates_path)
            self.loaders.append(loader)
            self.sysloaders.append(loader)

    def get_source(self, environment: Environment, template: str) -> Tuple[str, str, Callable]:
        if template.startswith('!'):
            loaders = self.sysloaders
            template = template[1:]
        else:
            loaders = self.loaders
        for loader in loaders:
            try:
                return loader.get_source(environment, template)
            except TemplateNotFound:
                pass
        raise TemplateNotFound(template)