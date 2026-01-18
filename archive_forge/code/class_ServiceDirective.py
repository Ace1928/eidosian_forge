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
class ServiceDirective(ObjectDescription):
    name = 'service'
    optional_arguments = 1

    def handle_signature(self, sig, signode):
        path = sig.split('/')
        namespace = '/'.join(path[:-1])
        if namespace and (not namespace.endswith('/')):
            namespace += '/'
        servicename = path[-1]
        if not namespace and (not servicename):
            servicename = '/'
        signode += addnodes.desc_annotation('service ', 'service ')
        if namespace:
            signode += addnodes.desc_addname(namespace, namespace)
        signode += addnodes.desc_name(servicename, servicename)
        return sig