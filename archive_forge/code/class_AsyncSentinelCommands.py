import warnings
class AsyncSentinelCommands(SentinelCommands):

    async def sentinel(self, *args) -> None:
        """Redis Sentinel's SENTINEL command."""
        super().sentinel(*args)