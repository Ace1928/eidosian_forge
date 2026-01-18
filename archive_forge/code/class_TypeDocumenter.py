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
class TypeDocumenter(autodoc.ClassDocumenter):
    objtype = 'type'
    directivetype = 'type'
    domain = 'wsme'
    required_arguments = 1
    default_samples_slot = 'after-docstring'
    option_spec = dict(autodoc.ClassDocumenter.option_spec, **{'protocols': lambda line: [v.strip() for v in line.split(',')], 'samples-slot': check_samples_slot})

    @staticmethod
    def can_document_member(member, membername, isattr, parent):
        return False

    def format_name(self):
        return self.object.__name__

    def format_signature(self):
        return u''

    def add_directive_header(self, sig):
        super(TypeDocumenter, self).add_directive_header(sig)
        result_len = len(self.directive.result)
        for index, item in zip(reversed(range(result_len)), reversed(self.directive.result)):
            if ':module:' in item:
                self.directive.result.pop(index)

    def import_object(self):
        if super(TypeDocumenter, self).import_object():
            wsme.types.register_type(self.object)
            return True
        else:
            return False

    def add_content(self, more_content):
        samples_slot = self.options.samples_slot or self.default_samples_slot

        def add_docstring():
            super(TypeDocumenter, self).add_content(more_content)

        def add_samples():
            protocols = get_protocols(self.options.protocols or self.env.app.config.wsme_protocols)
            content = []
            if protocols:
                sample_obj = make_sample_object(self.object)
                content.extend([_(u'Data samples:'), u'', u'.. cssclass:: toggle', u''])
                for name, protocol in protocols:
                    language, sample = protocol.encode_sample_value(self.object, sample_obj, format=True)
                    content.extend([name, u'    .. code-block:: ' + language, u''])
                    content.extend((u' ' * 8 + line for line in str(sample).split('\n')))
            for line in content:
                self.add_line(line, u'<wsmeext.sphinxext')
            self.add_line(u'', '<wsmeext.sphinxext>')
        if samples_slot == 'after-docstring':
            add_docstring()
            add_samples()
        elif samples_slot == 'before-docstring':
            add_samples()
            add_docstring()
        else:
            add_docstring()