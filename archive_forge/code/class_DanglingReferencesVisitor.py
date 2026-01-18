import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
class DanglingReferencesVisitor(nodes.SparseNodeVisitor):

    def __init__(self, document, unknown_reference_resolvers):
        nodes.SparseNodeVisitor.__init__(self, document)
        self.document = document
        self.unknown_reference_resolvers = unknown_reference_resolvers

    def unknown_visit(self, node):
        pass

    def visit_reference(self, node):
        if node.resolved or not node.hasattr('refname'):
            return
        refname = node['refname']
        id = self.document.nameids.get(refname)
        if id is None:
            for resolver_function in self.unknown_reference_resolvers:
                if resolver_function(node):
                    break
            else:
                if refname in self.document.nameids:
                    msg = self.document.reporter.error('Duplicate target name, cannot be used as a unique reference: "%s".' % node['refname'], base_node=node)
                else:
                    msg = self.document.reporter.error('Unknown target name: "%s".' % node['refname'], base_node=node)
                msgid = self.document.set_id(msg)
                prb = nodes.problematic(node.rawsource, node.rawsource, refid=msgid)
                try:
                    prbid = node['ids'][0]
                except IndexError:
                    prbid = self.document.set_id(prb)
                msg.add_backref(prbid)
                node.replace_self(prb)
        else:
            del node['refname']
            node['refid'] = id
            self.document.ids[id].note_referenced_by(id=id)
            node.resolved = 1
    visit_footnote_reference = visit_citation_reference = visit_reference