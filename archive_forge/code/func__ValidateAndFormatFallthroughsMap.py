from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import presentation_specs
import six
def _ValidateAndFormatFallthroughsMap(self, command_level_fallthroughs):
    """Validate formatting of fallthroughs and build map keyed to spec name."""
    spec_map = {}
    for key, fallthroughs_list in six.iteritems(command_level_fallthroughs):
        keys = key.split('.')
        if len(keys) != 2:
            raise ValueError('invalid fallthrough key: [{}]. Must be in format "FOO.a" where FOO is the presentation spec name and a is the attribute name.'.format(key))
        spec_name, attribute_name = keys
        self._ValidateSpecAndAttributeExist('key', spec_name, attribute_name)
        for fallthrough_string in fallthroughs_list:
            values = fallthrough_string.split('.')
            if len(values) not in [1, 2]:
                raise ValueError('invalid fallthrough value: [{}]. Must be in the form BAR.b or --baz'.format(fallthrough_string))
            if len(values) == 2:
                value_spec_name, value_attribute_name = values
                self._ValidateSpecAndAttributeExist('value', value_spec_name, value_attribute_name)
        spec_map.setdefault(spec_name, {})[attribute_name] = fallthroughs_list
    return spec_map