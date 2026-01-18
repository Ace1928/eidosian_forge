from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
class _Transform(object):
    """An object that contains an ordered list of _TransformCall objects.

  Attributes:
    _conditional: The resource_filter expression string for the if() transform.
    _transforms: The list of _TransformCall objects.
  """

    def __init__(self):
        self._conditional = None
        self._transforms = []

    def __str__(self):
        return '[{0}]'.format('.'.join(map(str, self._transforms)))

    @property
    def active(self):
        """The transform active level or None if always active."""
        return self._transforms[0].active if self._transforms else None

    @property
    def conditional(self):
        """The if() transform conditional expression string."""
        return self._conditional

    @property
    def global_restriction(self):
        """The global restriction string or None if not a global restriction.

    Terms in a fiter expression are sometimes called "restrictions" because
    they restrict or constrain values.  A regular restriction is of the form
    "attribute<op>operand".  A "global restriction" is a term that has no
    attribute or <op>.  It is a bare string that is matched against every
    attribute value in the resource object being filtered.  The global
    restriction matches if any of those values contains the string using case
    insensitive string match.

    Returns:
      The global restriction string or None if not a global restriction.
    """
        if len(self._transforms) != 1 or self._transforms[0].name != resource_projection_spec.GLOBAL_RESTRICTION_NAME:
            return None
        return self._transforms[0].args[0]

    @property
    def name(self):
        """The name of the last transform."""
        return self._transforms[-1].name if self._transforms else ''

    @property
    def term(self):
        """The first global restriction term."""
        return self._transforms[0].args[0] if self._transforms else ''

    def IsActive(self, active):
        """Returns True if the Transform active level is None or active."""
        return self._transforms and self.active in (None, active)

    def Add(self, transform):
        """Adds a transform to the list."""
        self._transforms.append(transform)

    def SetConditional(self, expr):
        """Sets the conditional expression string."""
        self._conditional = expr

    def Evaluate(self, obj, original_object=None):
        """Apply the list of transforms to obj and return the transformed value."""
        for transform in self._transforms:
            if transform.name == 'uri' and original_object is not None:
                obj = original_object
            if transform.map_transform and resource_property.IsListLike(obj):
                items = obj
                for _ in range(transform.map_transform - 1):
                    nested = []
                    try:
                        for item in items:
                            nested.extend(item)
                    except TypeError:
                        break
                    items = nested
                obj = []
                for item in items:
                    obj.append(transform.func(item, *transform.args, **transform.kwargs))
            elif obj or not transform.map_transform:
                obj = transform.func(obj, *transform.args, **transform.kwargs)
        return obj