import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
class NeptuneGraph(BaseNeptuneGraph):
    """Neptune wrapper for graph operations.

    Args:
        host: endpoint for the database instance
        port: port number for the database instance, default is 8182
        use_https: whether to use secure connection, default is True
        client: optional boto3 Neptune client
        credentials_profile_name: optional AWS profile name
        region_name: optional AWS region, e.g., us-west-2
        service: optional service name, default is neptunedata
        sign: optional, whether to sign the request payload, default is True

    Example:
        .. code-block:: python

        graph = NeptuneGraph(
            host='<my-cluster>',
            port=8182
        )

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

    def __init__(self, host: str, port: int=8182, use_https: bool=True, client: Any=None, credentials_profile_name: Optional[str]=None, region_name: Optional[str]=None, sign: bool=True) -> None:
        """Create a new Neptune graph wrapper instance."""
        try:
            if client is not None:
                self.client = client
            else:
                import boto3
                if credentials_profile_name is not None:
                    session = boto3.Session(profile_name=credentials_profile_name)
                else:
                    session = boto3.Session()
                client_params = {}
                if region_name:
                    client_params['region_name'] = region_name
                protocol = 'https' if use_https else 'http'
                client_params['endpoint_url'] = f'{protocol}://{host}:{port}'
                if sign:
                    self.client = session.client('neptunedata', **client_params)
                else:
                    from botocore import UNSIGNED
                    from botocore.config import Config
                    self.client = session.client('neptunedata', **client_params, config=Config(signature_version=UNSIGNED))
        except ImportError:
            raise ModuleNotFoundError('Could not import boto3 python package. Please install it with `pip install boto3`.')
        except Exception as e:
            if type(e).__name__ == 'UnknownServiceError':
                raise ModuleNotFoundError('NeptuneGraph requires a boto3 version 1.28.38 or greater.Please install it with `pip install -U boto3`.') from e
            else:
                raise ValueError('Could not load credentials to authenticate with AWS client. Please check that credentials in the specified profile name are valid.') from e
        try:
            self._refresh_schema()
        except Exception as e:
            raise NeptuneQueryException({'message': 'Could not get schema for Neptune database', 'detail': str(e)})

    def query(self, query: str, params: dict={}) -> Dict[str, Any]:
        """Query Neptune database."""
        try:
            return self.client.execute_open_cypher_query(openCypherQuery=query)['results']
        except Exception as e:
            raise NeptuneQueryException({'message': 'An error occurred while executing the query.', 'details': str(e)})

    def _get_summary(self) -> Dict:
        try:
            response = self.client.get_propertygraph_summary()
        except Exception as e:
            raise NeptuneQueryException({'message': 'Summary API is not available for this instance of Neptune,ensure the engine version is >=1.2.1.0', 'details': str(e)})
        try:
            summary = response['payload']['graphSummary']
        except Exception:
            raise NeptuneQueryException({'message': 'Summary API did not return a valid response.', 'details': response.content.decode()})
        else:
            return summary