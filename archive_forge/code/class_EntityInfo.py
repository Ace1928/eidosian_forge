import collections
import enum
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.autograph.pyct import templates
class EntityInfo(collections.namedtuple('EntityInfo', ('name', 'source_code', 'source_file', 'future_features', 'namespace'))):
    """Contains information about a Python entity.

  Immutable.

  Examples of entities include functions and classes.

  Attributes:
    name: The name that identifies this entity.
    source_code: The entity's source code.
    source_file: The entity's source file.
    future_features: Tuple[Text], the future features that this entity was
      compiled with. See
      https://docs.python.org/2/reference/simple_stmts.html#future.
    namespace: Dict[str, ], containing symbols visible to the entity (excluding
      parameters).
  """
    pass