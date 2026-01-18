import enum
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.util.tf_export import tf_export
@tf_export('autograph.experimental.Feature')
class Feature(enum.Enum):
    """This enumeration represents optional conversion options.

  These conversion options are experimental. They are subject to change without
  notice and offer no guarantees.

  _Example Usage_

  ```python
  optionals= tf.autograph.experimental.Feature.EQUALITY_OPERATORS
  @tf.function(experimental_autograph_options=optionals)
  def f(i):
    if i == 0:  # EQUALITY_OPERATORS allows the use of == here.
      tf.print('i is zero')
  ```

  Attributes:
    ALL: Enable all features.
    AUTO_CONTROL_DEPS: Insert of control dependencies in the generated code.
    ASSERT_STATEMENTS: Convert Tensor-dependent assert statements to tf.Assert.
    BUILTIN_FUNCTIONS: Convert builtin functions applied to Tensors to
      their TF counterparts.
    EQUALITY_OPERATORS: Whether to convert the equality operator ('==') to
      tf.math.equal.
    LISTS: Convert list idioms, like initializers, slices, append, etc.
    NAME_SCOPES: Insert name scopes that name ops according to context, like the
      function they were defined in.
  """
    ALL = 'ALL'
    AUTO_CONTROL_DEPS = 'AUTO_CONTROL_DEPS'
    ASSERT_STATEMENTS = 'ASSERT_STATEMENTS'
    BUILTIN_FUNCTIONS = 'BUILTIN_FUNCTIONS'
    EQUALITY_OPERATORS = 'EQUALITY_OPERATORS'
    LISTS = 'LISTS'
    NAME_SCOPES = 'NAME_SCOPES'

    @classmethod
    def all(cls):
        """Returns a tuple that enables all options."""
        return tuple(cls.__members__.values())

    @classmethod
    def all_but(cls, exclude):
        """Returns a tuple that enables all but the excluded options."""
        if not isinstance(exclude, (list, tuple, set)):
            exclude = (exclude,)
        return tuple(set(cls.all()) - set(exclude) - {cls.ALL})