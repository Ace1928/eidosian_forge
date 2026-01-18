from keystoneauth1.exceptions import base
class ConnectFailure(ConnectionError, RetriableConnectionFailure):
    message = 'Connection failure that may be retried.'