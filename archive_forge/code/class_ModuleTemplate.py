import json
import os
import re
import shutil
import stat
import tempfile
import types
import weakref
from mako import cache
from mako import codegen
from mako import compat
from mako import exceptions
from mako import runtime
from mako import util
from mako.lexer import Lexer
class ModuleTemplate(Template):
    """A Template which is constructed given an existing Python module.

    e.g.::

         t = Template("this is a template")
         f = file("mymodule.py", "w")
         f.write(t.code)
         f.close()

         import mymodule

         t = ModuleTemplate(mymodule)
         print(t.render())

    """

    def __init__(self, module, module_filename=None, template=None, template_filename=None, module_source=None, template_source=None, output_encoding=None, encoding_errors='strict', format_exceptions=False, error_handler=None, lookup=None, cache_args=None, cache_impl='beaker', cache_enabled=True, cache_type=None, cache_dir=None, cache_url=None, include_error_handler=None):
        self.module_id = re.sub('\\W', '_', module._template_uri)
        self.uri = module._template_uri
        self.input_encoding = module._source_encoding
        self.output_encoding = output_encoding
        self.encoding_errors = encoding_errors
        self.enable_loop = module._enable_loop
        self.module = module
        self.filename = template_filename
        ModuleInfo(module, module_filename, self, template_filename, module_source, template_source, module._template_uri)
        self.callable_ = self.module.render_body
        self.format_exceptions = format_exceptions
        self.error_handler = error_handler
        self.include_error_handler = include_error_handler
        self.lookup = lookup
        self._setup_cache_args(cache_impl, cache_enabled, cache_args, cache_type, cache_dir, cache_url)