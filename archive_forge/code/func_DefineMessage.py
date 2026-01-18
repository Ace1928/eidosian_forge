import pickle
import re
import sys
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def DefineMessage(self, module, name, children=None, add_to_module=True):
    """Define a new Message class in the context of a module.

        Used for easily describing complex Message hierarchy. Message
        is defined including all child definitions.

        Args:
          module: Fully qualified name of module to place Message class in.
          name: Name of Message to define within module.
          children: Define any level of nesting of children
            definitions. To define a message, map the name to another
            dictionary. The dictionary can itself contain additional
            definitions, and so on. To map to an Enum, define the Enum
            class separately and map it by name.
          add_to_module: If True, new Message class is added to
            module. If False, new Message is not added.

        """
    children = children or {}
    module_instance = self.DefineModule(module)
    for attribute, value in children.items():
        if isinstance(value, dict):
            children[attribute] = self.DefineMessage(module, attribute, value, False)
    children['__module__'] = module
    message_class = type(name, (messages.Message,), dict(children))
    if add_to_module:
        setattr(module_instance, name, message_class)
    return message_class