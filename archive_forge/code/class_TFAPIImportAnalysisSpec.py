import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
class TFAPIImportAnalysisSpec(ast_edits.APIAnalysisSpec):

    def __init__(self):
        self.symbols_to_detect = {}
        self.imports_to_detect = {('tensorflow', None): UnaliasedTFImport(), ('tensorflow.compat.v1', 'tf'): compat_v1_import, ('tensorflow.compat.v2', 'tf'): compat_v2_import}