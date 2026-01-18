import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
class UnaliasedTFImport(ast_edits.AnalysisResult):

    def __init__(self):
        self.log_level = ast_edits.ERROR
        self.log_message = 'The tf_upgrade_v2 script detected an unaliased `import tensorflow`. The script can only run when importing with `import tensorflow as tf`.'