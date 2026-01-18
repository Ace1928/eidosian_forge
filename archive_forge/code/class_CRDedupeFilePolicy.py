import base64
import functools
import itertools
import logging
import os
import queue
import random
import sys
import threading
import time
from types import TracebackType
from typing import (
import requests
import wandb
from wandb import util
from wandb.sdk.internal import internal_api
from ..lib import file_stream_utils
class CRDedupeFilePolicy(DefaultFilePolicy):
    """File stream policy for removing carriage-return erased characters.

    This is what a terminal does. We use it for console output to reduce the amount of
    data we need to send over the network (eg. for progress bars), while preserving the
    output's appearance in the web app.

    CR stands for "carriage return", for the character \\r. It tells the terminal to move
    the cursor back to the start of the current line. Progress bars (like tqdm) use \\r
    repeatedly to overwrite a line with newer updates. This gives the illusion of the
    progress bar filling up in real-time.
    """

    def __init__(self, start_chunk_id: int=0) -> None:
        super().__init__(start_chunk_id=start_chunk_id)
        self._prev_chunk = None
        self.global_offset = 0
        self.stderr = StreamCRState()
        self.stdout = StreamCRState()

    @staticmethod
    def get_consecutive_offsets(console: Dict[int, str]) -> List[List[int]]:
        """Compress consecutive line numbers into an interval.

        Args:
            console: Dict[int, str] which maps offsets (line numbers) to lines of text.
            It represents a mini version of our console dashboard on the UI.

        Returns:
            A list of intervals (we compress consecutive line numbers into an interval).

        Example:
            >>> console = {2: "", 3: "", 4: "", 5: "", 10: "", 11: "", 20: ""}
            >>> get_consecutive_offsets(console)
            [(2, 5), (10, 11), (20, 20)]
        """
        offsets = sorted(list(console.keys()))
        intervals: List = []
        for i, num in enumerate(offsets):
            if i == 0:
                intervals.append([num, num])
                continue
            largest = intervals[-1][1]
            if num == largest + 1:
                intervals[-1][1] = num
            else:
                intervals.append([num, num])
        return intervals

    @staticmethod
    def split_chunk(chunk: Chunk) -> Tuple[str, str]:
        """Split chunks.

        Args:
            chunk: object with two fields: filename (str) & data (str)
            `chunk.data` is a str containing the lines we want. It usually contains \\n or \\r or both.
            `chunk.data` has two possible formats (for the two streams - stdout and stderr):
                - "2020-08-25T20:38:36.895321 this is my line of text\\nsecond line\\n"
                - "ERROR 2020-08-25T20:38:36.895321 this is my line of text\\nsecond line\\nthird\\n".

                Here's another example with a carriage return \\r.
                - "ERROR 2020-08-25T20:38:36.895321 \\r progress bar\\n"

        Returns:
            A 2-tuple of strings.
            First str is prefix, either "ERROR {timestamp} " or "{timestamp} ".
            Second str is the rest of the string.

        Example:
            >>> chunk = Chunk(filename="output.log", data="ERROR 2020-08-25T20:38 this is my line of text\\n")
            >>> split_chunk(chunk)
            ("ERROR 2020-08-25T20:38 ", "this is my line of text\\n")
        """
        prefix = ''
        token, rest = chunk.data.split(' ', 1)
        if token == 'ERROR':
            prefix += token + ' '
            token, rest = rest.split(' ', 1)
        prefix += token + ' '
        return (prefix, rest)

    def process_chunks(self, chunks: List) -> List['ProcessedChunk']:
        """Process chunks.

        Args:
            chunks: List of Chunk objects. See description of chunk above in `split_chunk(...)`.

        Returns:
            List[Dict]. Each dict in the list contains two keys: an `offset` which holds the line number
            and `content` which maps to a list of consecutive lines starting from that offset.
            `offset` here means global line number in our console on the UI.

        Example:
            >>> chunks = [
                Chunk("output.log", "ERROR 2020-08-25T20:38 this is my line of text\\nboom\\n"),
                Chunk("output.log", "2020-08-25T20:38 this is test\\n"),
            ]
            >>> process_chunks(chunks)
            [
                {"offset": 0, "content": [
                    "ERROR 2020-08-25T20:38 this is my line of text\\n",
                    "ERROR 2020-08-25T20:38 boom\\n",
                    "2020-08-25T20:38 this is test\\n"
                    ]
                }
            ]
        """
        console = {}
        sep = os.linesep
        for c in chunks:
            prefix, logs_str = self.split_chunk(c)
            logs = logs_str.split(sep)
            for line in logs:
                stream = self.stderr if prefix.startswith('ERROR ') else self.stdout
                if line.startswith('\r'):
                    offset: int = stream.cr if stream.found_cr and stream.cr is not None else stream.last_normal or 0
                    stream.cr = offset
                    stream.found_cr = True
                    console[offset] = prefix + line[1:] + '\n'
                    if logs_str.count(sep) > 1 and logs_str.replace(sep, '').count('\r') == 1:
                        stream.found_cr = False
                elif line:
                    console[self.global_offset] = prefix + line + '\n'
                    stream.last_normal = self.global_offset
                    self.global_offset += 1
        intervals = self.get_consecutive_offsets(console)
        ret = []
        for a, b in intervals:
            processed_chunk: ProcessedChunk = {'offset': self._chunk_id + a, 'content': [console[i] for i in range(a, b + 1)]}
            ret.append(processed_chunk)
        return ret