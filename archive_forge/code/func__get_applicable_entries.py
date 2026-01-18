import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
def _get_applicable_entries(self, transformer_field, full_name, name):
    """Get all list entries indexed by name that apply to full_name or name."""
    function_transformers = getattr(self._api_change_spec, transformer_field, {})
    glob_name = '*.' + name if name else None
    transformers = []
    if full_name in function_transformers:
        transformers.append(function_transformers[full_name])
    if glob_name in function_transformers:
        transformers.append(function_transformers[glob_name])
    if '*' in function_transformers:
        transformers.append(function_transformers['*'])
    return transformers