import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class _CaseInsCasePresFilenameFeature(Feature):
    """Is the file-system case insensitive, but case-preserving?"""

    def _probe(self):
        fileno, name = tempfile.mkstemp(prefix='MixedCase')
        try:
            name = osutils.normpath(name)
            base, rel = osutils.split(name)
            found_rel = osutils.canonical_relpath(base, name)
            return found_rel == rel and os.path.isfile(name.upper()) and os.path.isfile(name.lower())
        finally:
            os.close(fileno)
            os.remove(name)

    def feature_name(self):
        return 'case-insensitive case-preserving filesystem'