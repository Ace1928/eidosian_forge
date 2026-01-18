import pathlib
from os import path
from pprint import pformat
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from jinja2 import BaseLoader, FileSystemLoader, TemplateNotFound
from jinja2.environment import Environment
from jinja2.sandbox import SandboxedEnvironment
from jinja2.utils import open_if_exists
from sphinx.application import TemplateBridge
from sphinx.theming import Theme
from sphinx.util import logging
from sphinx.util.osutil import mtimes_of_files
class SphinxFileSystemLoader(FileSystemLoader):
    """
    FileSystemLoader subclass that is not so strict about '..'  entries in
    template names.
    """

    def get_source(self, environment: Environment, template: str) -> Tuple[str, str, Callable]:
        for searchpath in self.searchpath:
            filename = str(pathlib.Path(searchpath, template))
            f = open_if_exists(filename)
            if f is None:
                continue
            with f:
                contents = f.read().decode(self.encoding)
            mtime = path.getmtime(filename)

            def uptodate() -> bool:
                try:
                    return path.getmtime(filename) == mtime
                except OSError:
                    return False
            return (contents, filename, uptodate)
        raise TemplateNotFound(template)