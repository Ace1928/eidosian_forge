import os
from mako.cache import CacheImpl
from mako.cache import register_plugin
from mako.template import Template
from .assertions import eq_
from .config import config
def _file_template(self, filename, **kw):
    filepath = self._file_path(filename)
    return Template(uri=filename, filename=filepath, module_directory=config.module_base, **kw)