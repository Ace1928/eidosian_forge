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
def _plnoun(self, word: str, count: Optional[Union[str, int]]=None) -> str:
    count = self.get_count(count)
    if count == 1:
        return word
    value = self.ud_match(word, self.pl_sb_user_defined)
    if value is not None:
        return value
    if word == '':
        return word
    word = Words(word)
    if word.last.lower() in pl_sb_uninflected_complete:
        if len(word.split_) >= 3:
            return self._handle_long_compounds(word, count=2) or word
        return word
    if word in pl_sb_uninflected_caps:
        return word
    for k, v in pl_sb_uninflected_bysize.items():
        if word.lowered[-k:] in v:
            return word
    if self.classical_dict['herd'] and word.last.lower() in pl_sb_uninflected_herd:
        return word
    mo = PL_SB_POSTFIX_ADJ_STEMS_RE.search(word)
    if mo and mo.group(2) != '':
        return f'{self._plnoun(mo.group(1), 2)}{mo.group(2)}'
    if ' a ' in word.lowered or '-a-' in word.lowered:
        mo = PL_SB_PREP_DUAL_COMPOUND_RE.search(word)
        if mo and mo.group(2) != '' and (mo.group(3) != ''):
            return f'{self._plnoun(mo.group(1), 2)}{mo.group(2)}{self._plnoun(mo.group(3))}'
    if len(word.split_) >= 3:
        handled_words = self._handle_long_compounds(word, count=2)
        if handled_words is not None:
            return handled_words
    mo = DENOMINATOR.search(word.lowered)
    if mo:
        index = len(mo.group('denominator'))
        return f'{self._plnoun(word[:index])}{word[index:]}'
    if len(word.split_) >= 2 and word.split_[-2] == 'degree':
        return ' '.join([self._plnoun(word.first)] + word.split_[1:])
    with contextlib.suppress(ValueError):
        return self._handle_prepositional_phrase(word.lowered, functools.partial(self._plnoun, count=2), '-')
    for k, v in pl_pron_acc_keys_bysize.items():
        if word.lowered[-k:] in v:
            for pk, pv in pl_prep_bysize.items():
                if word.lowered[:pk] in pv:
                    if word.lowered.split() == [word.lowered[:pk], word.lowered[-k:]]:
                        return word.lowered[:-k] + pl_pron_acc[word.lowered[-k:]]
    try:
        return pl_pron_nom[word.lowered]
    except KeyError:
        pass
    try:
        return pl_pron_acc[word.lowered]
    except KeyError:
        pass
    if word.last in pl_sb_irregular_caps:
        llen = len(word.last)
        return f'{word[:-llen]}{pl_sb_irregular_caps[word.last]}'
    lowered_last = word.last.lower()
    if lowered_last in pl_sb_irregular:
        llen = len(lowered_last)
        return f'{word[:-llen]}{pl_sb_irregular[lowered_last]}'
    dash_split = word.lowered.split('-')
    if ' '.join(dash_split[-2:]).lower() in pl_sb_irregular_compound:
        llen = len(' '.join(dash_split[-2:]))
        return f'{word[:-llen]}{pl_sb_irregular_compound[' '.join(dash_split[-2:]).lower()]}'
    if word.lowered[-3:] == 'quy':
        return f'{word[:-1]}ies'
    if word.lowered[-6:] == 'person':
        if self.classical_dict['persons']:
            return f'{word}s'
        else:
            return f'{word[:-4]}ople'
    if word.lowered[-3:] == 'man':
        for k, v in pl_sb_U_man_mans_bysize.items():
            if word.lowered[-k:] in v:
                return f'{word}s'
        for k, v in pl_sb_U_man_mans_caps_bysize.items():
            if word[-k:] in v:
                return f'{word}s'
        return f'{word[:-3]}men'
    if word.lowered[-5:] == 'mouse':
        return f'{word[:-5]}mice'
    if word.lowered[-5:] == 'louse':
        v = pl_sb_U_louse_lice_bysize.get(len(word))
        if v and word.lowered in v:
            return f'{word[:-5]}lice'
        return f'{word}s'
    if word.lowered[-5:] == 'goose':
        return f'{word[:-5]}geese'
    if word.lowered[-5:] == 'tooth':
        return f'{word[:-5]}teeth'
    if word.lowered[-4:] == 'foot':
        return f'{word[:-4]}feet'
    if word.lowered[-4:] == 'taco':
        return f'{word[:-5]}tacos'
    if word.lowered == 'die':
        return 'dice'
    if word.lowered[-4:] == 'ceps':
        return word
    if word.lowered[-4:] == 'zoon':
        return f'{word[:-2]}a'
    if word.lowered[-3:] in ('cis', 'sis', 'xis'):
        return f'{word[:-2]}es'
    for lastlet, d, numend, post in (('h', pl_sb_U_ch_chs_bysize, None, 's'), ('x', pl_sb_U_ex_ices_bysize, -2, 'ices'), ('x', pl_sb_U_ix_ices_bysize, -2, 'ices'), ('m', pl_sb_U_um_a_bysize, -2, 'a'), ('s', pl_sb_U_us_i_bysize, -2, 'i'), ('n', pl_sb_U_on_a_bysize, -2, 'a'), ('a', pl_sb_U_a_ae_bysize, None, 'e')):
        if word.lowered[-1] == lastlet:
            for k, v in d.items():
                if word.lowered[-k:] in v:
                    return word[:numend] + post
    if self.classical_dict['ancient']:
        if word.lowered[-4:] == 'trix':
            return f'{word[:-1]}ces'
        if word.lowered[-3:] in ('eau', 'ieu'):
            return f'{word}x'
        if word.lowered[-3:] in ('ynx', 'inx', 'anx') and len(word) > 4:
            return f'{word[:-1]}ges'
        for lastlet, d, numend, post in (('n', pl_sb_C_en_ina_bysize, -2, 'ina'), ('x', pl_sb_C_ex_ices_bysize, -2, 'ices'), ('x', pl_sb_C_ix_ices_bysize, -2, 'ices'), ('m', pl_sb_C_um_a_bysize, -2, 'a'), ('s', pl_sb_C_us_i_bysize, -2, 'i'), ('s', pl_sb_C_us_us_bysize, None, ''), ('a', pl_sb_C_a_ae_bysize, None, 'e'), ('a', pl_sb_C_a_ata_bysize, None, 'ta'), ('s', pl_sb_C_is_ides_bysize, -1, 'des'), ('o', pl_sb_C_o_i_bysize, -1, 'i'), ('n', pl_sb_C_on_a_bysize, -2, 'a')):
            if word.lowered[-1] == lastlet:
                for k, v in d.items():
                    if word.lowered[-k:] in v:
                        return word[:numend] + post
        for d, numend, post in ((pl_sb_C_i_bysize, None, 'i'), (pl_sb_C_im_bysize, None, 'im')):
            for k, v in d.items():
                if word.lowered[-k:] in v:
                    return word[:numend] + post
    if lowered_last in pl_sb_singular_s_complete:
        return f'{word}es'
    for k, v in pl_sb_singular_s_bysize.items():
        if word.lowered[-k:] in v:
            return f'{word}es'
    if word.lowered[-2:] == 'es' and word[0] == word[0].upper():
        return f'{word}es'
    if word.lowered[-1] == 'z':
        for k, v in pl_sb_z_zes_bysize.items():
            if word.lowered[-k:] in v:
                return f'{word}es'
        if word.lowered[-2:-1] != 'z':
            return f'{word}zes'
    if word.lowered[-2:] == 'ze':
        for k, v in pl_sb_ze_zes_bysize.items():
            if word.lowered[-k:] in v:
                return f'{word}s'
    if word.lowered[-2:] in ('ch', 'sh', 'zz', 'ss') or word.lowered[-1] == 'x':
        return f'{word}es'
    if word.lowered[-3:] in ('elf', 'alf', 'olf'):
        return f'{word[:-1]}ves'
    if word.lowered[-3:] == 'eaf' and word.lowered[-4:-3] != 'd':
        return f'{word[:-1]}ves'
    if word.lowered[-4:] in ('nife', 'life', 'wife'):
        return f'{word[:-2]}ves'
    if word.lowered[-3:] == 'arf':
        return f'{word[:-1]}ves'
    if word.lowered[-1] == 'y':
        if word.lowered[-2:-1] in 'aeiou' or len(word) == 1:
            return f'{word}s'
        if self.classical_dict['names']:
            if word.lowered[-1] == 'y' and word[0] == word[0].upper():
                return f'{word}s'
        return f'{word[:-1]}ies'
    if lowered_last in pl_sb_U_o_os_complete:
        return f'{word}s'
    for k, v in pl_sb_U_o_os_bysize.items():
        if word.lowered[-k:] in v:
            return f'{word}s'
    if word.lowered[-2:] in ('ao', 'eo', 'io', 'oo', 'uo'):
        return f'{word}s'
    if word.lowered[-1] == 'o':
        return f'{word}es'
    return f'{word}s'