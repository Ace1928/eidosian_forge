import inspect
import re
import sys
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain
from sphinx.domains import ObjType
from sphinx.domains.python import PyAttribute
from sphinx.domains.python import PyClasslike
from sphinx.domains.python import PyMethod
from sphinx.ext import autodoc
from sphinx.locale import _
from sphinx.roles import XRefRole
from sphinx.util.docfields import Field
from sphinx.util.nodes import make_refnode
import wsme
import wsme.rest.json
import wsme.rest.xml
import wsme.types
class RootDirective(Directive):
    """
    This directive is to tell what class is the Webservice root
    """
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {'webpath': directives.unchanged}

    def run(self):
        env = self.state.document.settings.env
        rootpath = self.arguments[0].strip()
        env.temp_data['wsme:rootpath'] = rootpath
        if 'wsme:root' in env.temp_data:
            del env.temp_data['wsme:root']
        if 'webpath' in self.options:
            env.temp_data['wsme:webpath'] = self.options['webpath']
        return []