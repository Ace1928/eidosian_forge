import ast
import collections
import contextlib
import functools
import itertools
import re
from numbers import Number
from typing import (
from more_itertools import windowed_complete
from typeguard import typechecked
from typing_extensions import Annotated, Literal
def _pl_check_plurals_N(self, word1: str, word2: str) -> bool:
    stem_endings = ((pl_sb_C_a_ata, 'as', 'ata'), (pl_sb_C_is_ides, 'is', 'ides'), (pl_sb_C_a_ae, 's', 'e'), (pl_sb_C_en_ina, 'ens', 'ina'), (pl_sb_C_um_a, 'ums', 'a'), (pl_sb_C_us_i, 'uses', 'i'), (pl_sb_C_on_a, 'ons', 'a'), (pl_sb_C_o_i_stems, 'os', 'i'), (pl_sb_C_ex_ices, 'exes', 'ices'), (pl_sb_C_ix_ices, 'ixes', 'ices'), (pl_sb_C_i, 's', 'i'), (pl_sb_C_im, 's', 'im'), ('.*eau', 's', 'x'), ('.*ieu', 's', 'x'), ('.*tri', 'xes', 'ces'), ('.{2,}[yia]n', 'xes', 'ges'))
    words = map(Words, (word1, word2))
    pair = '|'.join((word.last for word in words))
    return pair in pl_sb_irregular_s.values() or pair in pl_sb_irregular.values() or pair in pl_sb_irregular_caps.values() or any((self._pl_reg_plurals(pair, stems, end1, end2) for stems, end1, end2 in stem_endings))