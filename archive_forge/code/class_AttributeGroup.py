from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import six
class AttributeGroup(object):
    """Represents an object that gets transformed to an argument group.

  Attributes:
    concept: Concept, the underlying concept.
    key: str, the name by which the Attribute is looked up in the dependency
      view.
    attributes: [Attribute | AttributeGroup], the list of attributes or
      attribute groups contained in this attribute group.
    kwargs: {str: any}, other metadata describing the attribute. Available
      keys include: required (bool), mutex (bool), hidden (bool), help (str).
      **Note: This is currently used essentially as a passthrough to the
      argparse library.
  """

    def __init__(self, concept=None, attributes=None, **kwargs):
        self.concept = concept
        self.key = concept.key
        self.attributes = attributes or []
        self.kwargs = kwargs