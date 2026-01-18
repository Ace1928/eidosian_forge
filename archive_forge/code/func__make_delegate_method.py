import functools
def _make_delegate_method(attr_name):

    async def method(self, *args, **kwargs):
        cb = functools.partial(getattr(self._file, attr_name), *args, **kwargs)
        return await self._loop.run_in_executor(self._executor, cb)
    return method