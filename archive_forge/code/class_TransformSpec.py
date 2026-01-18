import sys
from collections import namedtuple
class TransformSpec:
    """
    Runtime transform specification base class.

    TransformSpec subclass objects used by `docutils.transforms.Transformer`.
    """

    def get_transforms(self):
        """Transforms required by this class.  Override in subclasses."""
        if self.default_transforms != ():
            import warnings
            warnings.warn('default_transforms attribute deprecated.\nUse get_transforms() method instead.', DeprecationWarning)
            return list(self.default_transforms)
        return []
    default_transforms = ()
    unknown_reference_resolvers = ()
    'List of functions to try to resolve unknown references.  Unknown\n    references have a \'refname\' attribute which doesn\'t correspond to any\n    target in the document.  Called when the transforms in\n    `docutils.tranforms.references` are unable to find a correct target.  The\n    list should contain functions which will try to resolve unknown\n    references, with the following signature::\n\n        def reference_resolver(node):\n            \'\'\'Returns boolean: true if resolved, false if not.\'\'\'\n\n    If the function is able to resolve the reference, it should also remove\n    the \'refname\' attribute and mark the node as resolved::\n\n        del node[\'refname\']\n        node.resolved = 1\n\n    Each function must have a "priority" attribute which will affect the order\n    the unknown_reference_resolvers are run::\n\n        reference_resolver.priority = 100\n\n    Override in subclasses.'