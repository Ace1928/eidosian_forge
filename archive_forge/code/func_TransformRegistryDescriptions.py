from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import pkgutil
import textwrap
from googlecloudsdk import api_lib
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import range  # pylint: disable=redefined-builtin
def TransformRegistryDescriptions():
    """Returns help markdown for all registered resource transforms."""
    descriptions = []
    apis = set([name for _, name, _ in pkgutil.iter_modules(api_lib.__path__) if name])
    for api in ['builtin'] + sorted(apis):
        transforms = _GetApiTransforms(api)
        if transforms:
            descriptions.append('\nThe {api} transform functions are:\n{desc}\n'.format(api=api, desc=TransformsDescriptions(transforms)))
    return ''.join(descriptions)