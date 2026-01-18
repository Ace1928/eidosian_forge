import sys
import os
import re
import time
from docutils import nodes, utils, languages
from docutils import ApplicationError, DataError
from docutils.transforms import Transform, TransformError
from docutils.transforms import parts, references, misc
class PEPZeroSpecial(nodes.SparseNodeVisitor):
    """
    Perform the special processing needed by PEP 0:

    - Mask email addresses.

    - Link PEP numbers in the second column of 4-column tables to the PEPs
      themselves.
    """
    pep_url = Headers.pep_url

    def unknown_visit(self, node):
        pass

    def visit_reference(self, node):
        node.replace_self(mask_email(node))

    def visit_field_list(self, node):
        if 'rfc2822' in node['classes']:
            raise nodes.SkipNode

    def visit_tgroup(self, node):
        self.pep_table = node['cols'] == 4
        self.entry = 0

    def visit_colspec(self, node):
        self.entry += 1
        if self.pep_table and self.entry == 2:
            node['classes'].append('num')

    def visit_row(self, node):
        self.entry = 0

    def visit_entry(self, node):
        self.entry += 1
        if self.pep_table and self.entry == 2 and (len(node) == 1):
            node['classes'].append('num')
            p = node[0]
            if isinstance(p, nodes.paragraph) and len(p) == 1:
                text = p.astext()
                try:
                    pep = int(text)
                    ref = self.document.settings.pep_base_url + self.pep_url % pep
                    p[0] = nodes.reference(text, text, refuri=ref)
                except ValueError:
                    pass