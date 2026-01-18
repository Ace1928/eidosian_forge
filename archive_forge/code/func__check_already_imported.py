import argparse
import collections
import importlib
import os
import sys
from tensorflow.python.tools.api.generator import doc_srcs
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export
import sys as _sys
from tensorflow.python.util import module_wrapper as _module_wrapper
def _check_already_imported(self, symbol_id, api_name):
    if api_name in self._dest_import_to_id and symbol_id != self._dest_import_to_id[api_name] and (symbol_id != -1):
        raise SymbolExposedTwiceError(f'Trying to export multiple symbols with same name: {api_name}')
    self._dest_import_to_id[api_name] = symbol_id