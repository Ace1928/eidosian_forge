from __future__ import annotations
import logging
import os
import shutil
import sys
import tempfile
from email.message import Message
from enum import IntEnum
from io import BytesIO
from numbers import Number
from typing import TYPE_CHECKING
from .decoders import Base64Decoder, QuotedPrintableDecoder
from .exceptions import FileError, FormParserError, MultipartParseError, QuerystringParseError
def _internal_write(self, data: bytes, length: int) -> int:
    boundary = self.boundary
    state = self.state
    index = self.index
    flags = self.flags
    i = 0

    def set_mark(name):
        self.marks[name] = i

    def delete_mark(name, reset=False):
        self.marks.pop(name, None)

    def data_callback(name, remaining=False):
        marked_index = self.marks.get(name)
        if marked_index is None:
            return
        if remaining:
            self.callback(name, data, marked_index, length)
            self.marks[name] = 0
        else:
            self.callback(name, data, marked_index, i)
            self.marks.pop(name, None)
    while i < length:
        c = data[i]
        if state == MultipartState.START:
            if c == CR or c == LF:
                i += 1
                self.logger.debug('Skipping leading CR/LF at %d', i)
                continue
            index = 0
            state = MultipartState.START_BOUNDARY
            i -= 1
        elif state == MultipartState.START_BOUNDARY:
            if index == len(boundary) - 2:
                if c != CR:
                    msg = 'Did not find CR at end of boundary (%d)' % (i,)
                    self.logger.warning(msg)
                    e = MultipartParseError(msg)
                    e.offset = i
                    raise e
                index += 1
            elif index == len(boundary) - 2 + 1:
                if c != LF:
                    msg = 'Did not find LF at end of boundary (%d)' % (i,)
                    self.logger.warning(msg)
                    e = MultipartParseError(msg)
                    e.offset = i
                    raise e
                index = 0
                self.callback('part_begin')
                state = MultipartState.HEADER_FIELD_START
            else:
                if c != boundary[index + 2]:
                    msg = 'Did not find boundary character %r at index %d' % (c, index + 2)
                    self.logger.warning(msg)
                    e = MultipartParseError(msg)
                    e.offset = i
                    raise e
                index += 1
        elif state == MultipartState.HEADER_FIELD_START:
            index = 0
            set_mark('header_field')
            state = MultipartState.HEADER_FIELD
            i -= 1
        elif state == MultipartState.HEADER_FIELD:
            if c == CR:
                delete_mark('header_field')
                state = MultipartState.HEADERS_ALMOST_DONE
                i += 1
                continue
            index += 1
            if c == HYPHEN:
                pass
            elif c == COLON:
                if index == 1:
                    msg = 'Found 0-length header at %d' % (i,)
                    self.logger.warning(msg)
                    e = MultipartParseError(msg)
                    e.offset = i
                    raise e
                data_callback('header_field')
                state = MultipartState.HEADER_VALUE_START
            else:
                cl = lower_char(c)
                if cl < LOWER_A or cl > LOWER_Z:
                    msg = 'Found non-alphanumeric character %r in header at %d' % (c, i)
                    self.logger.warning(msg)
                    e = MultipartParseError(msg)
                    e.offset = i
                    raise e
        elif state == MultipartState.HEADER_VALUE_START:
            if c == SPACE:
                i += 1
                continue
            set_mark('header_value')
            state = MultipartState.HEADER_VALUE
            i -= 1
        elif state == MultipartState.HEADER_VALUE:
            if c == CR:
                data_callback('header_value')
                self.callback('header_end')
                state = MultipartState.HEADER_VALUE_ALMOST_DONE
        elif state == MultipartState.HEADER_VALUE_ALMOST_DONE:
            if c != LF:
                msg = 'Did not find LF character at end of header (found %r)' % (c,)
                self.logger.warning(msg)
                e = MultipartParseError(msg)
                e.offset = i
                raise e
            state = MultipartState.HEADER_FIELD_START
        elif state == MultipartState.HEADERS_ALMOST_DONE:
            if c != LF:
                msg = f'Did not find LF at end of headers (found {c!r})'
                self.logger.warning(msg)
                e = MultipartParseError(msg)
                e.offset = i
                raise e
            self.callback('headers_finished')
            state = MultipartState.PART_DATA_START
        elif state == MultipartState.PART_DATA_START:
            set_mark('part_data')
            state = MultipartState.PART_DATA
            i -= 1
        elif state == MultipartState.PART_DATA:
            prev_index = index
            boundary_length = len(boundary)
            boundary_end = boundary_length - 1
            data_length = length
            boundary_chars = self.boundary_chars
            if index == 0:
                i += boundary_end
                while i < data_length - 1 and data[i] not in boundary_chars:
                    i += boundary_length
                i -= boundary_end
                c = data[i]
            if index < boundary_length:
                if boundary[index] == c:
                    if index == 0:
                        data_callback('part_data')
                    index += 1
                else:
                    index = 0
            elif index == boundary_length:
                index += 1
                if c == CR:
                    flags |= FLAG_PART_BOUNDARY
                elif c == HYPHEN:
                    flags |= FLAG_LAST_BOUNDARY
                else:
                    index = 0
            elif index == boundary_length + 1:
                if flags & FLAG_PART_BOUNDARY:
                    if c == LF:
                        flags &= ~FLAG_PART_BOUNDARY
                        self.callback('part_end')
                        self.callback('part_begin')
                        index = 0
                        state = MultipartState.HEADER_FIELD_START
                        i += 1
                        continue
                    index = 0
                    flags &= ~FLAG_PART_BOUNDARY
                elif flags & FLAG_LAST_BOUNDARY:
                    if c == HYPHEN:
                        self.callback('part_end')
                        self.callback('end')
                        state = MultipartState.END
                    else:
                        index = 0
            if index > 0:
                self.lookbehind[index - 1] = c
            elif prev_index > 0:
                lb_data = join_bytes(self.lookbehind)
                self.callback('part_data', lb_data, 0, prev_index)
                prev_index = 0
                set_mark('part_data')
                i -= 1
        elif state == MultipartState.END:
            if c not in (CR, LF):
                self.logger.warning("Consuming a byte '0x%x' in the end state", c)
        else:
            msg = 'Reached an unknown state %d at %d' % (state, i)
            self.logger.warning(msg)
            e = MultipartParseError(msg)
            e.offset = i
            raise e
        i += 1
    data_callback('header_field', True)
    data_callback('header_value', True)
    data_callback('part_data', True)
    self.state = state
    self.index = index
    self.flags = flags
    return length