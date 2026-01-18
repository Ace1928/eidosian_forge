import redis
from ...asyncio.client import Pipeline as AsyncioPipeline
from .commands import (
class AsyncSearch(Search, AsyncSearchCommands):

    class BatchIndexer(Search.BatchIndexer):
        """
        A batch indexer allows you to automatically batch
        document indexing in pipelines, flushing it every N documents.
        """

        async def add_document(self, doc_id, nosave=False, score=1.0, payload=None, replace=False, partial=False, no_create=False, **fields):
            """
            Add a document to the batch query
            """
            self.client._add_document(doc_id, conn=self._pipeline, nosave=nosave, score=score, payload=payload, replace=replace, partial=partial, no_create=no_create, **fields)
            self.current_chunk += 1
            self.total += 1
            if self.current_chunk >= self.chunk_size:
                await self.commit()

        async def commit(self):
            """
            Manually commit and flush the batch indexing query
            """
            await self._pipeline.execute()
            self.current_chunk = 0

    def pipeline(self, transaction=True, shard_hint=None):
        """Creates a pipeline for the SEARCH module, that can be used for executing
        SEARCH commands, as well as classic core commands.
        """
        p = AsyncPipeline(connection_pool=self.client.connection_pool, response_callbacks=self._MODULE_CALLBACKS, transaction=transaction, shard_hint=shard_hint)
        p.index_name = self.index_name
        return p