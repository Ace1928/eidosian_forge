import _sre
from . import _parser
from ._constants import *
from ._casefix import _EXTRA_CASES
def _compile_info(code, pattern, flags):
    lo, hi = pattern.getwidth()
    if hi > MAXCODE:
        hi = MAXCODE
    if lo == 0:
        code.extend([INFO, 4, 0, lo, hi])
        return
    prefix = []
    prefix_skip = 0
    charset = []
    if not (flags & SRE_FLAG_IGNORECASE and flags & SRE_FLAG_LOCALE):
        prefix, prefix_skip, got_all = _get_literal_prefix(pattern, flags)
        if not prefix:
            charset = _get_charset_prefix(pattern, flags)
    emit = code.append
    emit(INFO)
    skip = len(code)
    emit(0)
    mask = 0
    if prefix:
        mask = SRE_INFO_PREFIX
        if prefix_skip is None and got_all:
            mask = mask | SRE_INFO_LITERAL
    elif charset:
        mask = mask | SRE_INFO_CHARSET
    emit(mask)
    if lo < MAXCODE:
        emit(lo)
    else:
        emit(MAXCODE)
        prefix = prefix[:MAXCODE]
    emit(hi)
    if prefix:
        emit(len(prefix))
        if prefix_skip is None:
            prefix_skip = len(prefix)
        emit(prefix_skip)
        code.extend(prefix)
        code.extend(_generate_overlap_table(prefix))
    elif charset:
        charset, hascased = _optimize_charset(charset)
        assert not hascased
        _compile_charset(charset, flags, code)
    code[skip] = len(code) - skip