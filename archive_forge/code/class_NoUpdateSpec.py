import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
class NoUpdateSpec(APIChangeSpec):
    """A specification of an API change which doesn't change anything."""

    def __init__(self):
        self.function_handle = {}
        self.function_reorders = {}
        self.function_keyword_renames = {}
        self.symbol_renames = {}
        self.function_warnings = {}
        self.change_to_function = {}
        self.module_deprecations = {}
        self.function_transformers = {}
        self.import_renames = {}