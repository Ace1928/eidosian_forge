import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
def _maybe_add_warning(self, node, full_name):
    """Adds an error to be printed about full_name at node."""
    function_warnings = self._api_change_spec.function_warnings
    if full_name in function_warnings:
        level, message = function_warnings[full_name]
        message = message.replace('<function name>', full_name)
        self.add_log(level, node.lineno, node.col_offset, '%s requires manual check. %s' % (full_name, message))
        return True
    else:
        return False