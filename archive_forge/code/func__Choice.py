from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core.util import scaled_integer
import six
def _Choice(valid_choices):

    def _Parse(value):
        value = six.text_type(value.lower())
        if value not in valid_choices:
            raise arg_parsers.ArgumentTypeError('[type] must be one of [{0}]'.format(','.join(valid_choices)))
        return value
    return _Parse