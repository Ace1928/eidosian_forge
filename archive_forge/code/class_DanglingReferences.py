import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
class DanglingReferences(Transform):
    """
    Check for dangling references (incl. footnote & citation) and for
    unreferenced targets.
    """
    default_priority = 850

    def apply(self):
        visitor = DanglingReferencesVisitor(self.document, self.document.transformer.unknown_reference_resolvers)
        self.document.walk(visitor)
        for target in self.document.traverse(nodes.target):
            if not target.referenced:
                if target.get('anonymous'):
                    continue
                if target['names']:
                    naming = target['names'][0]
                elif target['ids']:
                    naming = target['ids'][0]
                else:
                    naming = target['refid']
                self.document.reporter.info('Hyperlink target "%s" is not referenced.' % naming, base_node=target)