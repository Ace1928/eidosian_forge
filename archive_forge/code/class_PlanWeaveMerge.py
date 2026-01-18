import itertools
import os
import struct
from copy import copy
from io import BytesIO
from typing import Any, Tuple
from zlib import adler32
from ..lazy_import import lazy_import
import fastbencode as bencode
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import graph as _mod_graph
from .. import osutils
from .. import transport as _mod_transport
from ..registry import Registry
from ..textmerge import TextMerge
from . import index
class PlanWeaveMerge(TextMerge):
    """Weave merge that takes a plan as its input.

    This exists so that VersionedFile.plan_merge is implementable.
    Most callers will want to use WeaveMerge instead.
    """

    def __init__(self, plan, a_marker=TextMerge.A_MARKER, b_marker=TextMerge.B_MARKER):
        TextMerge.__init__(self, a_marker, b_marker)
        self.plan = list(plan)

    def _merge_struct(self):
        lines_a = []
        lines_b = []
        ch_a = ch_b = False

        def outstanding_struct():
            if not lines_a and (not lines_b):
                return
            elif ch_a and (not ch_b):
                yield (lines_a,)
            elif ch_b and (not ch_a):
                yield (lines_b,)
            elif lines_a == lines_b:
                yield (lines_a,)
            else:
                yield (lines_a, lines_b)
        for state, line in self.plan:
            if state == 'unchanged':
                yield from outstanding_struct()
                lines_a = []
                lines_b = []
                ch_a = ch_b = False
            if state == 'unchanged':
                if line:
                    yield ([line],)
            elif state == 'killed-a':
                ch_a = True
                lines_b.append(line)
            elif state == 'killed-b':
                ch_b = True
                lines_a.append(line)
            elif state == 'new-a':
                ch_a = True
                lines_a.append(line)
            elif state == 'new-b':
                ch_b = True
                lines_b.append(line)
            elif state == 'conflicted-a':
                ch_b = ch_a = True
                lines_a.append(line)
            elif state == 'conflicted-b':
                ch_b = ch_a = True
                lines_b.append(line)
            elif state == 'killed-both':
                ch_b = ch_a = True
            elif state not in ('irrelevant', 'ghost-a', 'ghost-b', 'killed-base'):
                raise AssertionError(state)
        yield from outstanding_struct()

    def base_from_plan(self):
        """Construct a BASE file from the plan text."""
        base_lines = []
        for state, line in self.plan:
            if state in ('killed-a', 'killed-b', 'killed-both', 'unchanged'):
                base_lines.append(line)
            elif state not in ('killed-base', 'irrelevant', 'ghost-a', 'ghost-b', 'new-a', 'new-b', 'conflicted-a', 'conflicted-b'):
                raise AssertionError('Unknown state: {}'.format(state))
        return base_lines