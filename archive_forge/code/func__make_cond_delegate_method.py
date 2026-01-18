import functools
def _make_cond_delegate_method(attr_name):
    """For spooled temp files, delegate only if rolled to file object"""

    async def method(self, *args, **kwargs):
        if self._file._rolled:
            cb = functools.partial(getattr(self._file, attr_name), *args, **kwargs)
            return await self._loop.run_in_executor(self._executor, cb)
        else:
            return getattr(self._file, attr_name)(*args, **kwargs)
    return method