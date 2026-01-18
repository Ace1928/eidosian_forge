from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChildLink(_messages.Message):
    """Metadata associated with a parent-child relationship appearing in a
  PlanNode.

  Fields:
    childIndex: The node to which the link points.
    type: The type of the link. For example, in Hash Joins this could be used
      to distinguish between the build child and the probe child, or in the
      case of the child being an output variable, to represent the tag
      associated with the output variable.
    variable: Only present if the child node is SCALAR and corresponds to an
      output variable of the parent node. The field carries the name of the
      output variable. For example, a `TableScan` operator that reads rows
      from a table will have child links to the `SCALAR` nodes representing
      the output variables created for each column that is read by the
      operator. The corresponding `variable` fields will be set to the
      variable names assigned to the columns.
  """
    childIndex = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    type = _messages.StringField(2)
    variable = _messages.StringField(3)