from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_transform
import six
def _ParseKey(self):
    """Parses a key and optional attributes from the expression.

    The parsed key is appended to the ordered list of keys via _AddKey().
    Transform functions and key attributes are also handled here.

    Raises:
      ExpressionSyntaxError: The expression has a syntax error.
    """
    key, attribute = self._lex.KeyWithAttribute()
    if self._lex.IsCharacter('(', eoi_ok=True):
        add_transform = self._lex.Transform(key.pop(), self._projection.active)
    else:
        add_transform = None
    if not self.__key_attributes_only and attribute or (self.__key_attributes_only and attribute and (not key)):
        attribute = copy.copy(attribute)
    else:
        attribute = self._Attribute(self._projection.PROJECT)
    if not attribute.transform:
        attribute.transform = add_transform
    elif add_transform:
        attribute.transform._transforms.extend(add_transform._transforms)
    self._lex.SkipSpace()
    if self._lex.IsCharacter(':'):
        self._ParseKeyAttributes(key, attribute)
    if attribute.transform and attribute.transform.conditional:
        conditionals = self._projection.symbols.get(resource_transform.GetTypeDataName('conditionals'))

        def EvalGlobalRestriction(unused_obj, restriction, unused_pattern):
            return getattr(conditionals, restriction, None)
        defaults = resource_projection_spec.ProjectionSpec(symbols={resource_projection_spec.GLOBAL_RESTRICTION_NAME: EvalGlobalRestriction})
        if not resource_filter.Compile(attribute.transform.conditional, defaults=defaults).Evaluate(conditionals):
            return
    if attribute.label is None and (not key) and attribute.transform:
        attribute.label = self._AngrySnakeCase([attribute.transform.name] + attribute.transform._transforms[0].args)
    self._AddKey(key, attribute)