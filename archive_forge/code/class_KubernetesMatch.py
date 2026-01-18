from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KubernetesMatch(_messages.Message):
    """The scope of objects to which a given constraint will be applied

  Enums:
    ScopeValueValuesEnum: Matcher to match on scope of objects.

  Fields:
    excludedNamespaces: Matcher to match on objects not in excluded
      namespaces. Supports a prefix-based glob.
    groupKinds: Matcher to match on objects based on api group or kind.
    labelSelector: Matcher to match objects based on label keys or values.
    name: Matcher to match on an object's name. Supports a prefix-based glob.
    namespaceSelector: Matcher to match on namespace.
    namespaces: Matcher to match on objects in given namespaces. Supports a
      prefix-based glob.
    scope: Matcher to match on scope of objects.
  """

    class ScopeValueValuesEnum(_messages.Enum):
        """Matcher to match on scope of objects.

    Values:
      SCOPE_UNSPECIFIED: Unspecified scope.
      SCOPE_ALL: Scope `*`, match all resources.
      SCOPE_CLUSTER: Scope `Cluster`, match cluster-scoped resources.
      SCOPE_NAMESPACED: Scope `Namespaced`, match namespace-scoped resources.
    """
        SCOPE_UNSPECIFIED = 0
        SCOPE_ALL = 1
        SCOPE_CLUSTER = 2
        SCOPE_NAMESPACED = 3
    excludedNamespaces = _messages.StringField(1, repeated=True)
    groupKinds = _messages.MessageField('GroupKind', 2, repeated=True)
    labelSelector = _messages.StringField(3)
    name = _messages.StringField(4)
    namespaceSelector = _messages.StringField(5)
    namespaces = _messages.StringField(6, repeated=True)
    scope = _messages.EnumField('ScopeValueValuesEnum', 7)