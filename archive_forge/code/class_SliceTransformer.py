import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import templates
class SliceTransformer(converter.Base):
    """Converts slicing operations to their TF counterpart.

  Currently, relying on the default slice operator that Tensor uses is
  insufficient, because TensorArray and tensor lists use dedicated index read
  and write functions.
  """

    def _process_single_assignment(self, target, value):
        if not isinstance(target, gast.Subscript):
            return None
        s = target.slice
        if isinstance(s, (gast.Tuple, gast.Slice)):
            return None
        template = '\n      target = ag__.set_item(target, key, item)\n    '
        return templates.replace(template, target=target.value, key=target.slice, item=value)

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if len(node.targets) != 1:
            raise NotImplementedError('multiple assignment')
        replacement = self._process_single_assignment(node.targets[0], node.value)
        if replacement is not None:
            return replacement
        return node

    def visit_Subscript(self, node):
        node = self.generic_visit(node)
        s = node.slice
        if isinstance(s, (gast.Tuple, gast.Slice)):
            return node
        if not isinstance(node.ctx, gast.Load):
            return node
        dtype = self.get_definition_directive(node.value, directives.set_element_type, 'dtype', default=templates.replace_as_expression('None'))
        template = '\n      ag__.get_item(\n          target,\n          key,\n          opts=ag__.GetItemOpts(element_dtype=dtype))\n    '
        return templates.replace_as_expression(template, target=node.value, key=s, dtype=dtype)