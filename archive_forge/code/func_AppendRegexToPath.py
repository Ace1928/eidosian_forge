from __future__ import absolute_import
import re
def AppendRegexToPath(path, regex):
    """Equivalent to os.path.join(), except uses forward slashes always."""
    return path.rstrip('/') + '/' + regex