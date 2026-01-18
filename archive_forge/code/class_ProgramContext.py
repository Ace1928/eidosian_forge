import enum
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.util.tf_export import tf_export
class ProgramContext(object):
    """ProgramContext keeps track of converting function hierarchies.

  Attributes:
    options: ConversionOptions
    autograph_module: Deprecated. Do not use.
  """

    def __init__(self, options, autograph_module=None):
        self.options = options
        self.autograph_module = autograph_module