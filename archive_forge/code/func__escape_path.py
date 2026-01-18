from copy import copy
from ..osutils import contains_linebreaks, contains_whitespace, sha_strings
from ..tree import Tree
def _escape_path(self, path):
    if contains_linebreaks(path):
        raise ValueError(path)
    if not isinstance(path, str):
        path = path.decode('ascii')
    if path == '':
        path = '.'
    return path.replace('\\', '/').replace(' ', '\\ ')