from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def Scalar(self, event, loader):
    """Handle scalar value

    Since scalars are simple values that are passed directly in by the
    parser, handle like any value with no additional processing.

    Of course, key values will be handles specially.  A key value is recognized
    when the top token is _TOKEN_MAPPING.

    Args:
      event: Event containing scalar value.
    """
    self._HandleAnchor(event)
    if event.tag is None and self._top[0] != _TOKEN_MAPPING:
        try:
            tag = loader.resolve(yaml.nodes.ScalarNode, event.value, event.implicit)
        except IndexError:
            tag = loader.DEFAULT_SCALAR_TAG
    else:
        tag = event.tag
    if tag is None:
        value = event.value
    else:
        node = yaml.nodes.ScalarNode(tag, event.value, event.start_mark, event.end_mark, event.style)
        value = loader.construct_object(node)
    self._HandleValue(value)