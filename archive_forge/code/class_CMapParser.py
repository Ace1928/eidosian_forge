import gzip
import logging
import os
import os.path
import pickle as pickle
import struct
import sys
from typing import (
from .encodingdb import name2unicode
from .psparser import KWD
from .psparser import PSEOF
from .psparser import PSKeyword
from .psparser import PSLiteral
from .psparser import PSStackParser
from .psparser import PSSyntaxError
from .psparser import literal_name
from .utils import choplist
from .utils import nunpack
class CMapParser(PSStackParser[PSKeyword]):

    def __init__(self, cmap: CMapBase, fp: BinaryIO) -> None:
        PSStackParser.__init__(self, fp)
        self.cmap = cmap
        self._in_cmap = True
        self._warnings: Set[str] = set()
        return

    def run(self) -> None:
        try:
            self.nextobject()
        except PSEOF:
            pass
        return
    KEYWORD_BEGINCMAP = KWD(b'begincmap')
    KEYWORD_ENDCMAP = KWD(b'endcmap')
    KEYWORD_USECMAP = KWD(b'usecmap')
    KEYWORD_DEF = KWD(b'def')
    KEYWORD_BEGINCODESPACERANGE = KWD(b'begincodespacerange')
    KEYWORD_ENDCODESPACERANGE = KWD(b'endcodespacerange')
    KEYWORD_BEGINCIDRANGE = KWD(b'begincidrange')
    KEYWORD_ENDCIDRANGE = KWD(b'endcidrange')
    KEYWORD_BEGINCIDCHAR = KWD(b'begincidchar')
    KEYWORD_ENDCIDCHAR = KWD(b'endcidchar')
    KEYWORD_BEGINBFRANGE = KWD(b'beginbfrange')
    KEYWORD_ENDBFRANGE = KWD(b'endbfrange')
    KEYWORD_BEGINBFCHAR = KWD(b'beginbfchar')
    KEYWORD_ENDBFCHAR = KWD(b'endbfchar')
    KEYWORD_BEGINNOTDEFRANGE = KWD(b'beginnotdefrange')
    KEYWORD_ENDNOTDEFRANGE = KWD(b'endnotdefrange')

    def do_keyword(self, pos: int, token: PSKeyword) -> None:
        """ToUnicode CMaps

        See Section 5.9.2 - ToUnicode CMaps of the PDF Reference.
        """
        if token is self.KEYWORD_BEGINCMAP:
            self._in_cmap = True
            self.popall()
            return
        elif token is self.KEYWORD_ENDCMAP:
            self._in_cmap = False
            return
        if not self._in_cmap:
            return
        if token is self.KEYWORD_DEF:
            try:
                (_, k), (_, v) = self.pop(2)
                self.cmap.set_attr(literal_name(k), v)
            except PSSyntaxError:
                pass
            return
        if token is self.KEYWORD_USECMAP:
            try:
                (_, cmapname), = self.pop(1)
                self.cmap.use_cmap(CMapDB.get_cmap(literal_name(cmapname)))
            except PSSyntaxError:
                pass
            except CMapDB.CMapNotFound:
                pass
            return
        if token is self.KEYWORD_BEGINCODESPACERANGE:
            self.popall()
            return
        if token is self.KEYWORD_ENDCODESPACERANGE:
            self.popall()
            return
        if token is self.KEYWORD_BEGINCIDRANGE:
            self.popall()
            return
        if token is self.KEYWORD_ENDCIDRANGE:
            objs = [obj for __, obj in self.popall()]
            for start_byte, end_byte, cid in choplist(3, objs):
                if not isinstance(start_byte, bytes):
                    self._warn_once('The start object of begincidrange is not a byte.')
                    continue
                if not isinstance(end_byte, bytes):
                    self._warn_once('The end object of begincidrange is not a byte.')
                    continue
                if not isinstance(cid, int):
                    self._warn_once('The cid object of begincidrange is not a byte.')
                    continue
                if len(start_byte) != len(end_byte):
                    self._warn_once('The start and end byte of begincidrange have different lengths.')
                    continue
                start_prefix = start_byte[:-4]
                end_prefix = end_byte[:-4]
                if start_prefix != end_prefix:
                    self._warn_once('The prefix of the start and end byte of begincidrange are not the same.')
                    continue
                svar = start_byte[-4:]
                evar = end_byte[-4:]
                start = nunpack(svar)
                end = nunpack(evar)
                vlen = len(svar)
                for i in range(end - start + 1):
                    x = start_prefix + struct.pack('>L', start + i)[-vlen:]
                    self.cmap.add_cid2unichr(cid + i, x)
            return
        if token is self.KEYWORD_BEGINCIDCHAR:
            self.popall()
            return
        if token is self.KEYWORD_ENDCIDCHAR:
            objs = [obj for __, obj in self.popall()]
            for cid, code in choplist(2, objs):
                if isinstance(code, bytes) and isinstance(cid, int):
                    self.cmap.add_cid2unichr(cid, code)
            return
        if token is self.KEYWORD_BEGINBFRANGE:
            self.popall()
            return
        if token is self.KEYWORD_ENDBFRANGE:
            objs = [obj for __, obj in self.popall()]
            for start_byte, end_byte, code in choplist(3, objs):
                if not isinstance(start_byte, bytes):
                    self._warn_once('The start object is not a byte.')
                    continue
                if not isinstance(end_byte, bytes):
                    self._warn_once('The end object is not a byte.')
                    continue
                if len(start_byte) != len(end_byte):
                    self._warn_once('The start and end byte have different lengths.')
                    continue
                start = nunpack(start_byte)
                end = nunpack(end_byte)
                if isinstance(code, list):
                    if len(code) != end - start + 1:
                        self._warn_once('The difference between the start and end offsets does not match the code length.')
                    for cid, unicode_value in zip(range(start, end + 1), code):
                        self.cmap.add_cid2unichr(cid, unicode_value)
                else:
                    assert isinstance(code, bytes)
                    var = code[-4:]
                    base = nunpack(var)
                    prefix = code[:-4]
                    vlen = len(var)
                    for i in range(end - start + 1):
                        x = prefix + struct.pack('>L', base + i)[-vlen:]
                        self.cmap.add_cid2unichr(start + i, x)
            return
        if token is self.KEYWORD_BEGINBFCHAR:
            self.popall()
            return
        if token is self.KEYWORD_ENDBFCHAR:
            objs = [obj for __, obj in self.popall()]
            for cid, code in choplist(2, objs):
                if isinstance(cid, bytes) and isinstance(code, bytes):
                    self.cmap.add_cid2unichr(nunpack(cid), code)
            return
        if token is self.KEYWORD_BEGINNOTDEFRANGE:
            self.popall()
            return
        if token is self.KEYWORD_ENDNOTDEFRANGE:
            self.popall()
            return
        self.push((pos, token))

    def _warn_once(self, msg: str) -> None:
        """Warn once for each unique message"""
        if msg not in self._warnings:
            self._warnings.add(msg)
            base_msg = 'Ignoring (part of) ToUnicode map because the PDF data does not conform to the format. This could result in (cid) values in the output. '
            log.warning(base_msg + msg)