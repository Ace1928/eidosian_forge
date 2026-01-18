import inspect
from collections import UserList
from functools import partial
from itertools import islice, tee, zip_longest
from typing import Any, Callable
from kombu.utils.functional import LRUCache, dictfilter, is_list, lazy, maybe_evaluate, maybe_list, memoize
from vine import promise
from celery.utils.log import get_logger
class _regen(UserList, list):

    def __init__(self, it):
        self.__it = it
        self.__consumed = []
        self.__done = False

    def __reduce__(self):
        return (list, (self.data,))

    def map(self, func):
        self.__consumed = [func(el) for el in self.__consumed]
        self.__it = map(func, self.__it)

    def __length_hint__(self):
        return self.__it.__length_hint__()

    def __lookahead_consume(self, limit=None):
        if not self.__done and (limit is None or limit > 0):
            it = iter(self.__it)
            try:
                now = next(it)
            except StopIteration:
                return
            self.__consumed.append(now)
            while not self.__done:
                try:
                    next_ = next(it)
                    self.__consumed.append(next_)
                except StopIteration:
                    self.__done = True
                    break
                finally:
                    yield now
                now = next_
                if limit is not None:
                    limit -= 1
                    if limit <= 0:
                        break

    def __iter__(self):
        yield from self.__consumed
        yield from self.__lookahead_consume()

    def __getitem__(self, index):
        if index < 0:
            return self.data[index]
        consume_count = index - len(self.__consumed) + 1
        for _ in self.__lookahead_consume(limit=consume_count):
            pass
        return self.__consumed[index]

    def __bool__(self):
        if len(self.__consumed):
            return True
        try:
            next(iter(self))
        except StopIteration:
            return False
        else:
            return True

    @property
    def data(self):
        if not self.__done:
            self.__consumed.extend(self.__it)
            self.__done = True
        return self.__consumed

    def __repr__(self):
        return '<{}: [{}{}]>'.format(self.__class__.__name__, ', '.join((repr(e) for e in self.__consumed)), '...' if not self.__done else '')