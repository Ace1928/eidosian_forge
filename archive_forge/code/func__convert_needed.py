import sys
import logging
import os
import copy
from lib2to3.pgen2.parse import ParseError
from lib2to3.refactor import RefactoringTool
from libfuturize import fixes
def _convert_needed(self):
    fullname = self.name
    if any((fullname.startswith(path) for path in self.exclude_paths)):
        convert = False
    elif any((fullname.startswith(path) for path in self.include_paths)):
        convert = True
    else:
        convert = False
    return convert