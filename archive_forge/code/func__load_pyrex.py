imports, including parts of the standard library and installed
import glob
import imp
import os
import sys
from zipimport import zipimporter, ZipImportError
def _load_pyrex(name, filename):
    """Load a pyrex file given a name and filename."""