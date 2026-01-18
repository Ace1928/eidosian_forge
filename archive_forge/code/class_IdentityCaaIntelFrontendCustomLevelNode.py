from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityCaaIntelFrontendCustomLevelNode(_messages.Message):
    """Custom Level Node Tree for the Logical Expression Tree NextTAG: 4

  Enums:
    NodeTypeValueValuesEnum: Node type, indicate if it's an expression or
      AND/OR/NOT Logical Operator Node

  Fields:
    nodeId: Node id, used to map to its NodeValue and Troubleshooting metadata
    nodeType: Node type, indicate if it's an expression or AND/OR/NOT Logical
      Operator Node
    nodes: Child nodes
  """

    class NodeTypeValueValuesEnum(_messages.Enum):
        """Node type, indicate if it's an expression or AND/OR/NOT Logical
    Operator Node

    Values:
      CUSTOM_LEVEL_NODE_UNSPECIFIED: Reserved
      CUSTOM_LEVEL_NODE_EXPRESSION: Custom level Expression node
      CUSTOM_LEVEL_NODE_AND: Custom level AND node
      CUSTOM_LEVEL_NODE_OR: Custom level OR node
      CUSTOM_LEVEL_NODE_NOT: Custom level NOT node
    """
        CUSTOM_LEVEL_NODE_UNSPECIFIED = 0
        CUSTOM_LEVEL_NODE_EXPRESSION = 1
        CUSTOM_LEVEL_NODE_AND = 2
        CUSTOM_LEVEL_NODE_OR = 3
        CUSTOM_LEVEL_NODE_NOT = 4
    nodeId = _messages.IntegerField(1)
    nodeType = _messages.EnumField('NodeTypeValueValuesEnum', 2)
    nodes = _messages.MessageField('IdentityCaaIntelFrontendCustomLevelNode', 3, repeated=True)