from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
def ReplaceTemplateParams(node, params):
    """Apply the params provided into the template.

  Args:
    node: A node in the parsed template
    params: a dict of params of param-name -> param-value

  Returns:
    A tuple of (new_node, missing_params, used_params) where new_node is the
    node with all params replaced, missing_params is set of param
    references found in the node that were not provided and used_params were
    the params that we actually used.
  """
    if isinstance(node, six.string_types):
        if node.startswith('{{') and node.endswith('}}'):
            param = node[2:-2].strip()
            if param in params:
                return (params[param], set(), set([param]))
            else:
                return (node, set([param]), set())
    if isinstance(node, dict):
        missing_params = set()
        used_params = set()
        for k in node.keys():
            new_subnode, new_missing, new_used = ReplaceTemplateParams(node[k], params)
            node[k] = new_subnode
            missing_params.update(new_missing)
            used_params.update(new_used)
        return (node, missing_params, used_params)
    if isinstance(node, list):
        missing_params = set()
        used_params = set()
        new_node = []
        for subnode in node:
            new_subnode, new_missing, new_used = ReplaceTemplateParams(subnode, params)
            new_node.append(new_subnode)
            missing_params.update(new_missing)
            used_params.update(new_used)
        return (new_node, missing_params, used_params)
    return (node, set(), set())