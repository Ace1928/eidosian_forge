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
def _AddKey(self, key, attribute_add):
    """Propagates default attribute values and adds key to the projection.

    Args:
      key: The parsed key to add.
      attribute_add: Parsed _Attribute to add.
    """
    projection = self._root
    for name in key[:-1]:
        tree = projection.tree
        if name in tree:
            attribute = tree[name].attribute
            if attribute.flag != self._projection.PROJECT:
                attribute.flag = self._projection.INNER
        else:
            tree[name] = self._Tree(self._Attribute(self._projection.INNER))
        projection = tree[name]
    tree = projection.tree
    name = key[-1] if key else ''
    name_in_tree = name in tree
    if name_in_tree:
        attribute = tree[name].attribute
        if not self.__key_attributes_only and any([col for col in self._projection.Columns() if col.key == key]):
            attribute = copy.copy(attribute)
        if not self.__key_attributes_only or not attribute.order:
            attribute.hidden = False
    elif isinstance(name, six.integer_types) and None in tree:
        tree[name] = copy.deepcopy(tree[None])
        attribute = tree[name].attribute
    else:
        attribute = attribute_add
        if self.__key_attributes_only and attribute.order:
            attribute.hidden = True
        if key or attribute.transform:
            tree[name] = self._Tree(attribute)
    if attribute_add.order is not None:
        attribute.order = attribute_add.order
        if self.__key_attributes_only:
            self.__key_order_offset += 1
            attribute.skip_reorder = True
    if attribute_add.label is not None:
        attribute.label = attribute_add.label
    elif attribute.label is None:
        attribute.label = self._AngrySnakeCase(key)
    if attribute_add.align != resource_projection_spec.ALIGN_DEFAULT:
        attribute.align = attribute_add.align
    if attribute_add.optional is not None:
        attribute.optional = attribute_add.optional
    elif attribute.optional is None:
        attribute.optional = False
    if attribute_add.reverse is not None:
        attribute.reverse = attribute_add.reverse
    elif attribute.reverse is None:
        attribute.reverse = False
    if attribute_add.transform:
        attribute.transform = attribute_add.transform
    if attribute_add.subformat:
        attribute.subformat = attribute_add.subformat
    if attribute_add.width is not None:
        attribute.width = attribute_add.width
    elif attribute.width is None:
        attribute.width = False
    if attribute_add.wrap is not None:
        attribute.wrap = attribute_add.wrap
    elif attribute.wrap is None:
        attribute.wrap = False
    self._projection.AddAlias(attribute.label, key, attribute)
    if not self.__key_attributes_only or attribute.hidden:
        attribute.flag = self._projection.PROJECT
        self._projection.AddKey(key, attribute)
    elif not name_in_tree:
        attribute.flag = self._projection.DEFAULT