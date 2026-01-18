from html import escape as html_escape
from os.path import exists, isfile, splitext, abspath, join, isdir
from os import walk, sep, fsdecode
from IPython.core.display import DisplayObject, TextDisplayObject
from typing import Tuple, Iterable, Optional
def _format_path(self):
    result_lines = []
    if self.recursive:
        walked_dir = list(walk(self.path))
    else:
        walked_dir = [next(walk(self.path))]
    walked_dir.sort()
    for dirname, subdirs, fnames in walked_dir:
        result_lines += self.notebook_display_formatter(dirname, fnames, self.included_suffixes)
    return '\n'.join(result_lines)