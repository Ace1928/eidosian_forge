import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
def _plan_annotate_merge(annotated_a, annotated_b, ancestors_a, ancestors_b):

    def status_a(revision, text):
        if revision in ancestors_b:
            return ('killed-b', text)
        else:
            return ('new-a', text)

    def status_b(revision, text):
        if revision in ancestors_a:
            return ('killed-a', text)
        else:
            return ('new-b', text)
    plain_a = [t for a, t in annotated_a]
    plain_b = [t for a, t in annotated_b]
    matcher = patiencediff.PatienceSequenceMatcher(None, plain_a, plain_b)
    blocks = matcher.get_matching_blocks()
    a_cur = 0
    b_cur = 0
    for ai, bi, l in blocks:
        for revision, text in annotated_a[a_cur:ai]:
            yield status_a(revision, text)
        for revision, text in annotated_b[b_cur:bi]:
            yield status_b(revision, text)
        a_cur = ai + l
        b_cur = bi + l
        for text_a in plain_a[ai:a_cur]:
            yield ('unchanged', text_a)