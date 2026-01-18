from aiokeydb.v1.exceptions import ResponseError, DataError
from aiokeydb.v1.commands.graph.exceptions import VersionMismatchException
from aiokeydb.v1.commands.graph.execution_plan import ExecutionPlan
from aiokeydb.v1.commands.graph.query_result import AsyncQueryResult, QueryResult
class AsyncGraphCommands(GraphCommands):

    async def query(self, q, params=None, timeout=None, read_only=False, profile=False):
        """
        Executes a query against the graph.
        For more information see `GRAPH.QUERY <https://oss.redis.com/redisgraph/master/commands/#graphquery>`_. # noqa

        Args:

        q : str
            The query.
        params : dict
            Query parameters.
        timeout : int
            Maximum runtime for read queries in milliseconds.
        read_only : bool
            Executes a readonly query if set to True.
        profile : bool
            Return details on results produced by and time
            spent in each operation.
        """
        query = q
        query = self._build_params_header(params) + query
        if profile:
            cmd = PROFILE_CMD
        else:
            cmd = RO_QUERY_CMD if read_only else QUERY_CMD
        command = [cmd, self.name, query, '--compact']
        if isinstance(timeout, int):
            command.extend(['timeout', timeout])
        elif timeout is not None:
            raise Exception('Timeout argument must be a positive integer')
        try:
            response = await self.execute_command(*command)
            return await AsyncQueryResult().initialize(self, response, profile)
        except ResponseError as e:
            if 'unknown command' in str(e) and read_only:
                return await self.query(q, params, timeout, read_only=False)
            raise e
        except VersionMismatchException as e:
            self.version = e.version
            self._refresh_schema()
            return await self.query(q, params, timeout, read_only)

    async def execution_plan(self, query, params=None):
        """
        Get the execution plan for given query,
        GRAPH.EXPLAIN returns an array of operations.

        Args:
            query: the query that will be executed
            params: query parameters
        """
        query = self._build_params_header(params) + query
        plan = await self.execute_command(EXPLAIN_CMD, self.name, query)
        if isinstance(plan[0], bytes):
            plan = [b.decode() for b in plan]
        return '\n'.join(plan)

    async def explain(self, query, params=None):
        """
        Get the execution plan for given query,
        GRAPH.EXPLAIN returns ExecutionPlan object.

        Args:
            query: the query that will be executed
            params: query parameters
        """
        query = self._build_params_header(params) + query
        plan = await self.execute_command(EXPLAIN_CMD, self.name, query)
        return ExecutionPlan(plan)

    async def flush(self):
        """
        Commit the graph and reset the edges and the nodes to zero length.
        """
        await self.commit()
        self.nodes = {}
        self.edges = []