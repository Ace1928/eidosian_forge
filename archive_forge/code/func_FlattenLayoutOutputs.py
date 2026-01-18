from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from six.moves import range  # pylint: disable=redefined-builtin
def FlattenLayoutOutputs(manifest_layout):
    """Takes the layout from a manifest and returns the flattened outputs.

  List output 'foo: [A,B]' becomes 'foo[0]: A, foo[1]: B'
  Dict output 'bar: {a:1, b:2}' becomes 'bar[a]: 1, bar[b]: 2'
  Lists and Dicts whose values are themselves lists or dicts are not expanded.

  Args:
    manifest_layout: The 'layout' field from the manifest.

  Returns:
    A list of {'name': X, 'finalValue': Y} dicts built out of the 'outputs'
    section of the layout.
  """
    layout = yaml.load(manifest_layout)
    if not isinstance(layout, dict) or 'outputs' not in layout:
        return []
    outputs = []
    basic_outputs = layout['outputs']
    for basic_output in basic_outputs:
        if 'finalValue' not in basic_output or 'name' not in basic_output:
            continue
        name = basic_output['name']
        value = basic_output['finalValue']
        if isinstance(value, list):
            for pos in range(len(value)):
                final_name = '%s[%d]' % (name, pos)
                outputs.append(_BuildOutput(final_name, value[pos]))
        elif isinstance(value, dict):
            for key in value:
                final_name = '%s[%s]' % (name, key)
                outputs.append(_BuildOutput(final_name, value[key]))
        else:
            outputs.append(_BuildOutput(name, value))
    return outputs