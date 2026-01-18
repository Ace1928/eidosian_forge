from html import escape as html_escape
from os.path import exists, isfile, splitext, abspath, join, isdir
from os import walk, sep, fsdecode
from IPython.core.display import DisplayObject, TextDisplayObject
from typing import Tuple, Iterable, Optional
def _get_display_formatter(self, dirname_output_format, fname_output_format, fp_format, fp_cleaner=None):
    """generate built-in formatter function

        this is used to define both the notebook and terminal built-in
         formatters as they only differ by some wrapper text for each entry

        dirname_output_format: string to use for formatting directory
         names, dirname will be substituted for a single "%s" which
         must appear in this string
        fname_output_format: string to use for formatting file names,
         if a single "%s" appears in the string, fname will be substituted
         if two "%s" appear in the string, the path to fname will be
          substituted for the first and fname will be substituted for the
          second
        fp_format: string to use for formatting filepaths, must contain
         exactly two "%s" and the dirname will be substituted for the first
         and fname will be substituted for the second
        """

    def f(dirname, fnames, included_suffixes=None):
        result = []
        display_fnames = []
        for fname in fnames:
            if isfile(join(dirname, fname)) and (included_suffixes is None or splitext(fname)[1] in included_suffixes):
                display_fnames.append(fname)
        if len(display_fnames) == 0:
            pass
        else:
            dirname_output_line = dirname_output_format % dirname
            result.append(dirname_output_line)
            for fname in display_fnames:
                fp = fp_format % (dirname, fname)
                if fp_cleaner is not None:
                    fp = fp_cleaner(fp)
                try:
                    fname_output_line = fname_output_format % (fp, fname)
                except TypeError:
                    fname_output_line = fname_output_format % fname
                result.append(fname_output_line)
        return result
    return f