import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
class ExternalTargets(Transform):
    """
    Given::

        <paragraph>
            <reference refname="direct external">
                direct external
        <target id="id1" name="direct external" refuri="http://direct">

    The "refname" attribute is replaced by the direct "refuri" attribute::

        <paragraph>
            <reference refuri="http://direct">
                direct external
        <target id="id1" name="direct external" refuri="http://direct">
    """
    default_priority = 640

    def apply(self):
        for target in self.document.traverse(nodes.target):
            if target.hasattr('refuri'):
                refuri = target['refuri']
                for name in target['names']:
                    reflist = self.document.refnames.get(name, [])
                    if reflist:
                        target.note_referenced_by(name=name)
                    for ref in reflist:
                        if ref.resolved:
                            continue
                        del ref['refname']
                        ref['refuri'] = refuri
                        ref.resolved = 1