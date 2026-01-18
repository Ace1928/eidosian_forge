import enum
import gast
class Basic(NoValue):
    """Container for basic annotation keys.

  The enum values are used strictly for documentation purposes.
  """
    QN = 'Qualified name, as it appeared in the code. See qual_names.py.'
    SKIP_PROCESSING = 'This node should be preserved as is and not processed any further.'
    INDENT_BLOCK_REMAINDER = 'When a node is annotated with this, the remainder of the block should be indented below it. The annotation contains a tuple (new_body, name_map), where `new_body` is the new indented block and `name_map` allows renaming symbols.'
    ORIGIN = 'Information about the source code that converted code originated from. See origin_information.py.'
    DIRECTIVES = 'User directives associated with a statement or a variable. Typically, they affect the immediately-enclosing statement.'
    EXTRA_LOOP_TEST = 'A special annotation containing additional test code to be executed in for loops.'