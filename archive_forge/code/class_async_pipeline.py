from typing import TYPE_CHECKING
class async_pipeline:

    def __init__(self, keydb_obj: 'AsyncKeyDB'):
        self.p: 'AsyncPipeline' = keydb_obj.pipeline()

    async def __aenter__(self) -> 'AsyncPipeline':
        return self.p

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.p.execute()
        del self.p