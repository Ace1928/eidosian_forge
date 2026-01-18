import dataclasses
import socket
import ssl
import threading
import typing as t
def _iter_next(self) -> t.Iterator[MessageType]:
    idx = 0
    while True:
        with self._condition:
            if self._exp:
                raise Exception(f'Exception from receiving task: {self._exp}') from self._exp
            if idx < len(self._results):
                value = self._results[idx]
                idx += 1
                yield value
            else:
                self._condition.wait()