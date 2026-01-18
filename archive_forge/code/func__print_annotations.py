import sys
import time
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from . import config, errors, osutils
from .repository import _strip_NULL_ghosts
from .revision import CURRENT_REVISION, Revision
def _print_annotations(annotation, verbose, to_file, full, encoding):
    """Print annotations to to_file.

    :param to_file: The file to output the annotation to.
    :param verbose: Show all details rather than truncating to ensure
        reasonable text width.
    :param full: XXXX Not sure what this does.
    """
    if len(annotation) == 0:
        max_origin_len = max_revno_len = 0
    else:
        max_origin_len = max((len(x[1]) for x in annotation))
        max_revno_len = max((len(x[0]) for x in annotation))
    if not verbose:
        max_revno_len = min(max_revno_len, 12)
    max_revno_len = max(max_revno_len, 3)
    prevanno = ''
    for revno_str, author, date_str, line_rev_id, text in annotation:
        if verbose:
            anno = '%-*s %-*s %8s ' % (max_revno_len, revno_str, max_origin_len, author, date_str)
        else:
            if len(revno_str) > max_revno_len:
                revno_str = revno_str[:max_revno_len - 1] + '>'
            anno = '%-*s %-7s ' % (max_revno_len, revno_str, author[:7])
        if anno.lstrip() == '' and full:
            anno = prevanno
        to_file.write(anno)
        to_file.write('| {}\n'.format(text.decode(encoding)))
        prevanno = anno