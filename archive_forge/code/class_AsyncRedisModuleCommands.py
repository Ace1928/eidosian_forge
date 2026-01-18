from json import JSONDecoder, JSONEncoder
class AsyncRedisModuleCommands(RedisModuleCommands):

    def ft(self, index_name='idx'):
        """Access the search namespace, providing support for redis search."""
        from .search import AsyncSearch
        s = AsyncSearch(client=self, index_name=index_name)
        return s

    def graph(self, index_name='idx'):
        """Access the graph namespace, providing support for
        redis graph data.
        """
        from .graph import AsyncGraph
        g = AsyncGraph(client=self, name=index_name)
        return g