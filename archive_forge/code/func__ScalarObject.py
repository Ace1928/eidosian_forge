from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from ruamel import yaml
import six
def _ScalarObject(self, node, value, raw=False):
    if not isinstance(node, yaml.nodes.ScalarNode):
        raise yaml.constructor.ConstructorError(None, None, 'expected a scalar node, but found {}'.format(node.id), node.start_mark)
    ret_val = node.value if raw else self._ScalarType(node)
    ret_val.lc = yaml.comments.LineCol()
    ret_val.lc.line = node.start_mark.line
    ret_val.lc.col = node.start_mark.column
    ret_val.value = value
    return ret_val