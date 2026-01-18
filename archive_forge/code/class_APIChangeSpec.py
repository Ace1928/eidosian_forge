import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
class APIChangeSpec:
    """This class defines the transformations that need to happen.

  This class must provide the following fields:

  * `function_keyword_renames`: maps function names to a map of old -> new
    argument names
  * `symbol_renames`: maps function names to new function names
  * `change_to_function`: a set of function names that have changed (for
    notifications)
  * `function_reorders`: maps functions whose argument order has changed to the
    list of arguments in the new order
  * `function_warnings`: maps full names of functions to warnings that will be
    printed out if the function is used. (e.g. tf.nn.convolution())
  * `function_transformers`: maps function names to custom handlers
  * `module_deprecations`: maps module names to warnings that will be printed
    if the module is still used after all other transformations have run
  * `import_renames`: maps import name (must be a short name without '.')
    to ImportRename instance.

  For an example, see `TFAPIChangeSpec`.
  """

    def preprocess(self, root_node):
        """Preprocess a parse tree. Return a preprocessed node, logs and errors."""
        return (root_node, [], [])

    def clear_preprocessing(self):
        """Restore this APIChangeSpec to before it preprocessed a file.

    This is needed if preprocessing a file changed any rewriting rules.
    """
        pass