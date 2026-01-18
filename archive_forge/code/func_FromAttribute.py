from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.command_lib.concepts import base
from googlecloudsdk.command_lib.concepts import exceptions
from googlecloudsdk.command_lib.concepts import names
import six
@classmethod
def FromAttribute(cls, attribute):
    """Builds the dependency tree from the attribute."""
    kwargs = {'concept': attribute.concept}
    marshal = attribute.concept.Marshal()
    if marshal:
        attributes = [concept.Attribute() for concept in marshal]
    elif not isinstance(attribute, base.Attribute):
        attributes = attribute.attributes
    else:
        attributes = None
    if isinstance(attribute, base.Attribute) and (marshal or not attributes):
        kwargs['arg_name'] = attribute.arg_name
        kwargs['fallthroughs'] = attribute.fallthroughs
    if attributes:
        kwargs['dependencies'] = {a.concept.key: DependencyNode.FromAttribute(a) for a in attributes}
    return DependencyNode(attribute.concept.key, not isinstance(attribute, base.Attribute), **kwargs)