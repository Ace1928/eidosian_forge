from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import presentation_specs
import six
def _AddSpec(self, presentation_spec):
    """Adds a given presentation spec to the concept holder's spec registry.

    Args:
      presentation_spec: PresentationSpec, the spec to be added.

    Raises:
      ValueError: if two presentation specs have the same name, if two
        presentation specs are both positional, or if two args are going to
        overlap.
    """
    for spec_name in self._specs:
        if self._ArgNameMatches(spec_name, presentation_spec.name):
            raise ValueError('Attempted to add two concepts with the same name: [{}, {}]'.format(spec_name, presentation_spec.name))
        if util.IsPositional(spec_name) and util.IsPositional(presentation_spec.name):
            raise ValueError('Attempted to add multiple concepts with positional arguments: [{}, {}]'.format(spec_name, presentation_spec.name))
    for a, arg_name in six.iteritems(presentation_spec.attribute_to_args_map):
        del a
        name = util.NormalizeFormat(arg_name)
        if name in self._all_args:
            raise ValueError('Attempted to add a duplicate argument name: [{}]'.format(arg_name))
        self._all_args.append(name)
    self._specs[presentation_spec.name] = presentation_spec