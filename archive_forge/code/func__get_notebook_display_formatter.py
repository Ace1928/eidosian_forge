from html import escape as html_escape
from os.path import exists, isfile, splitext, abspath, join, isdir
from os import walk, sep, fsdecode
from IPython.core.display import DisplayObject, TextDisplayObject
from typing import Tuple, Iterable, Optional
def _get_notebook_display_formatter(self, spacer='&nbsp;&nbsp;'):
    """ generate function to use for notebook formatting
        """
    dirname_output_format = self.result_html_prefix + '%s/' + self.result_html_suffix
    fname_output_format = self.result_html_prefix + spacer + self.html_link_str + self.result_html_suffix
    fp_format = self.url_prefix + '%s/%s'
    if sep == '\\':

        def fp_cleaner(fp):
            return fp.replace('\\', '/')
    else:
        fp_cleaner = None
    return self._get_display_formatter(dirname_output_format, fname_output_format, fp_format, fp_cleaner)