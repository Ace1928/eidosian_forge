from types import coroutine
from collections.abc import Coroutine
from asyncio import get_running_loop
class AsyncIndirectBase(AsyncBase):

    def __init__(self, name, loop, executor, indirect):
        self._indirect = indirect
        self._name = name
        super().__init__(None, loop, executor)

    @property
    def _file(self):
        return self._indirect()

    @_file.setter
    def _file(self, v):
        pass