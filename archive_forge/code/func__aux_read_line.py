import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
def _aux_read_line(self):
    line = self.file.readline()
    if isinstance(line, bytes):
        return line.decode(self.encoding)
    return line