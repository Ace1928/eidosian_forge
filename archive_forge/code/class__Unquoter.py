import codecs
import re
from string import ascii_letters, ascii_lowercase, digits
from typing import Optional, cast
class _Unquoter:

    def __init__(self, *, unsafe: str='', qs: bool=False) -> None:
        self._unsafe = unsafe
        self._qs = qs
        self._quoter = _Quoter()
        self._qs_quoter = _Quoter(qs=True)

    def __call__(self, val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        if not isinstance(val, str):
            raise TypeError('Argument should be str')
        if not val:
            return ''
        decoder = cast(codecs.BufferedIncrementalDecoder, utf8_decoder())
        ret = []
        idx = 0
        while idx < len(val):
            ch = val[idx]
            idx += 1
            if ch == '%' and idx <= len(val) - 2:
                pct = val[idx:idx + 2]
                if _IS_HEX_STR.fullmatch(pct):
                    b = bytes([int(pct, base=16)])
                    idx += 2
                    try:
                        unquoted = decoder.decode(b)
                    except UnicodeDecodeError:
                        start_pct = idx - 3 - len(decoder.buffer) * 3
                        ret.append(val[start_pct:idx - 3])
                        decoder.reset()
                        try:
                            unquoted = decoder.decode(b)
                        except UnicodeDecodeError:
                            ret.append(val[idx - 3:idx])
                            continue
                    if not unquoted:
                        continue
                    if self._qs and unquoted in '+=&;':
                        to_add = self._qs_quoter(unquoted)
                        if to_add is None:
                            raise RuntimeError('Cannot quote None')
                        ret.append(to_add)
                    elif unquoted in self._unsafe:
                        to_add = self._quoter(unquoted)
                        if to_add is None:
                            raise RuntimeError('Cannot quote None')
                        ret.append(to_add)
                    else:
                        ret.append(unquoted)
                    continue
            if decoder.buffer:
                start_pct = idx - 1 - len(decoder.buffer) * 3
                ret.append(val[start_pct:idx - 1])
                decoder.reset()
            if ch == '+':
                if not self._qs or ch in self._unsafe:
                    ret.append('+')
                else:
                    ret.append(' ')
                continue
            if ch in self._unsafe:
                ret.append('%')
                h = hex(ord(ch)).upper()[2:]
                for ch in h:
                    ret.append(ch)
                continue
            ret.append(ch)
        if decoder.buffer:
            ret.append(val[-len(decoder.buffer) * 3:])
        ret2 = ''.join(ret)
        if ret2 == val:
            return val
        return ret2