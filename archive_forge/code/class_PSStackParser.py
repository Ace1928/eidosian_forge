import logging
import re
from typing import (
from . import settings
from .utils import choplist
class PSStackParser(PSBaseParser, Generic[ExtraT]):

    def __init__(self, fp: BinaryIO) -> None:
        PSBaseParser.__init__(self, fp)
        self.reset()
        return

    def reset(self) -> None:
        self.context: List[Tuple[int, Optional[str], List[PSStackEntry[ExtraT]]]] = []
        self.curtype: Optional[str] = None
        self.curstack: List[PSStackEntry[ExtraT]] = []
        self.results: List[PSStackEntry[ExtraT]] = []
        return

    def seek(self, pos: int) -> None:
        PSBaseParser.seek(self, pos)
        self.reset()
        return

    def push(self, *objs: PSStackEntry[ExtraT]) -> None:
        self.curstack.extend(objs)
        return

    def pop(self, n: int) -> List[PSStackEntry[ExtraT]]:
        objs = self.curstack[-n:]
        self.curstack[-n:] = []
        return objs

    def popall(self) -> List[PSStackEntry[ExtraT]]:
        objs = self.curstack
        self.curstack = []
        return objs

    def add_results(self, *objs: PSStackEntry[ExtraT]) -> None:
        try:
            log.debug('add_results: %r', objs)
        except Exception:
            log.debug('add_results: (unprintable object)')
        self.results.extend(objs)
        return

    def start_type(self, pos: int, type: str) -> None:
        self.context.append((pos, self.curtype, self.curstack))
        self.curtype, self.curstack = (type, [])
        log.debug('start_type: pos=%r, type=%r', pos, type)
        return

    def end_type(self, type: str) -> Tuple[int, List[PSStackType[ExtraT]]]:
        if self.curtype != type:
            raise PSTypeError('Type mismatch: {!r} != {!r}'.format(self.curtype, type))
        objs = [obj for _, obj in self.curstack]
        pos, self.curtype, self.curstack = self.context.pop()
        log.debug('end_type: pos=%r, type=%r, objs=%r', pos, type, objs)
        return (pos, objs)

    def do_keyword(self, pos: int, token: PSKeyword) -> None:
        return

    def nextobject(self) -> PSStackEntry[ExtraT]:
        """Yields a list of objects.

        Arrays and dictionaries are represented as Python lists and
        dictionaries.

        :return: keywords, literals, strings, numbers, arrays and dictionaries.
        """
        while not self.results:
            pos, token = self.nexttoken()
            if isinstance(token, (int, float, bool, str, bytes, PSLiteral)):
                self.push((pos, token))
            elif token == KEYWORD_ARRAY_BEGIN:
                self.start_type(pos, 'a')
            elif token == KEYWORD_ARRAY_END:
                try:
                    self.push(self.end_type('a'))
                except PSTypeError:
                    if settings.STRICT:
                        raise
            elif token == KEYWORD_DICT_BEGIN:
                self.start_type(pos, 'd')
            elif token == KEYWORD_DICT_END:
                try:
                    pos, objs = self.end_type('d')
                    if len(objs) % 2 != 0:
                        error_msg = 'Invalid dictionary construct: %r' % objs
                        raise PSSyntaxError(error_msg)
                    d = {literal_name(k): v for k, v in choplist(2, objs) if v is not None}
                    self.push((pos, d))
                except PSTypeError:
                    if settings.STRICT:
                        raise
            elif token == KEYWORD_PROC_BEGIN:
                self.start_type(pos, 'p')
            elif token == KEYWORD_PROC_END:
                try:
                    self.push(self.end_type('p'))
                except PSTypeError:
                    if settings.STRICT:
                        raise
            elif isinstance(token, PSKeyword):
                log.debug('do_keyword: pos=%r, token=%r, stack=%r', pos, token, self.curstack)
                self.do_keyword(pos, token)
            else:
                log.error('unknown token: pos=%r, token=%r, stack=%r', pos, token, self.curstack)
                self.do_keyword(pos, token)
                raise
            if self.context:
                continue
            else:
                self.flush()
        obj = self.results.pop(0)
        try:
            log.debug('nextobject: %r', obj)
        except Exception:
            log.debug('nextobject: (unprintable object)')
        return obj