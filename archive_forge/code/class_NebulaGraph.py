import logging
from string import Template
from typing import Any, Dict, Optional
class NebulaGraph:
    """NebulaGraph wrapper for graph operations.

    NebulaGraph inherits methods from Neo4jGraph to bring ease to the user space.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    def __init__(self, space: str, username: str='root', password: str='nebula', address: str='127.0.0.1', port: int=9669, session_pool_size: int=30) -> None:
        """Create a new NebulaGraph wrapper instance."""
        try:
            import nebula3
            import pandas
        except ImportError:
            raise ValueError('Please install NebulaGraph Python client and pandas first: `pip install nebula3-python pandas`')
        self.username = username
        self.password = password
        self.address = address
        self.port = port
        self.space = space
        self.session_pool_size = session_pool_size
        self.session_pool = self._get_session_pool()
        self.schema = ''
        try:
            self.refresh_schema()
        except Exception as e:
            raise ValueError(f'Could not refresh schema. Error: {e}')

    def _get_session_pool(self) -> Any:
        assert all([self.username, self.password, self.address, self.port, self.space]), 'Please provide all of the following parameters: username, password, address, port, space'
        from nebula3.Config import SessionPoolConfig
        from nebula3.Exception import AuthFailedException, InValidHostname
        from nebula3.gclient.net.SessionPool import SessionPool
        config = SessionPoolConfig()
        config.max_size = self.session_pool_size
        try:
            session_pool = SessionPool(self.username, self.password, self.space, [(self.address, self.port)])
        except InValidHostname:
            raise ValueError('Could not connect to NebulaGraph database. Please ensure that the address and port are correct')
        try:
            session_pool.init(config)
        except AuthFailedException:
            raise ValueError('Could not connect to NebulaGraph database. Please ensure that the username and password are correct')
        except RuntimeError as e:
            raise ValueError(f'Error initializing session pool. Error: {e}')
        return session_pool

    def __del__(self) -> None:
        try:
            self.session_pool.close()
        except Exception as e:
            logger.warning(f'Could not close session pool. Error: {e}')

    @property
    def get_schema(self) -> str:
        """Returns the schema of the NebulaGraph database"""
        return self.schema

    def execute(self, query: str, params: Optional[dict]=None, retry: int=0) -> Any:
        """Query NebulaGraph database."""
        from nebula3.Exception import IOErrorException, NoValidSessionException
        from nebula3.fbthrift.transport.TTransport import TTransportException
        params = params or {}
        try:
            result = self.session_pool.execute_parameter(query, params)
            if not result.is_succeeded():
                logger.warning(f'Error executing query to NebulaGraph. Error: {result.error_msg()}\nQuery: {query} \n')
            return result
        except NoValidSessionException:
            logger.warning(f'No valid session found in session pool. Please consider increasing the session pool size. Current size: {self.session_pool_size}')
            raise ValueError(f'No valid session found in session pool. Please consider increasing the session pool size. Current size: {self.session_pool_size}')
        except RuntimeError as e:
            if retry < RETRY_TIMES:
                retry += 1
                logger.warning(f'Error executing query to NebulaGraph. Retrying ({retry}/{RETRY_TIMES})...\nquery: {query} \nError: {e}')
                return self.execute(query, params, retry)
            else:
                raise ValueError(f'Error executing query to NebulaGraph. Error: {e}')
        except (TTransportException, IOErrorException):
            if retry < RETRY_TIMES:
                retry += 1
                logger.warning(f'Connection issue with NebulaGraph. Retrying ({retry}/{RETRY_TIMES})...\n to recreate session pool')
                self.session_pool = self._get_session_pool()
                return self.execute(query, params, retry)

    def refresh_schema(self) -> None:
        """
        Refreshes the NebulaGraph schema information.
        """
        tags_schema, edge_types_schema, relationships = ([], [], [])
        for tag in self.execute('SHOW TAGS').column_values('Name'):
            tag_name = tag.cast()
            tag_schema = {'tag': tag_name, 'properties': []}
            r = self.execute(f'DESCRIBE TAG `{tag_name}`')
            props, types = (r.column_values('Field'), r.column_values('Type'))
            for i in range(r.row_size()):
                tag_schema['properties'].append((props[i].cast(), types[i].cast()))
            tags_schema.append(tag_schema)
        for edge_type in self.execute('SHOW EDGES').column_values('Name'):
            edge_type_name = edge_type.cast()
            edge_schema = {'edge': edge_type_name, 'properties': []}
            r = self.execute(f'DESCRIBE EDGE `{edge_type_name}`')
            props, types = (r.column_values('Field'), r.column_values('Type'))
            for i in range(r.row_size()):
                edge_schema['properties'].append((props[i].cast(), types[i].cast()))
            edge_types_schema.append(edge_schema)
            r = self.execute(rel_query.substitute(edge_type=edge_type_name)).column_values('rels')
            if len(r) > 0:
                relationships.append(r[0].cast())
        self.schema = f'Node properties: {tags_schema}\nEdge properties: {edge_types_schema}\nRelationships: {relationships}\n'

    def query(self, query: str, retry: int=0) -> Dict[str, Any]:
        result = self.execute(query, retry=retry)
        columns = result.keys()
        d: Dict[str, list] = {}
        for col_num in range(result.col_size()):
            col_name = columns[col_num]
            col_list = result.column_values(col_name)
            d[col_name] = [x.cast() for x in col_list]
        return d