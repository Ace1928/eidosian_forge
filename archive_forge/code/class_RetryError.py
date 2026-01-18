import logging
from wandb_graphql import parse, introspection_query, build_ast_schema, build_client_schema
from wandb_graphql.validation import validate
from .transport.local_schema import LocalSchemaTransport
class RetryError(Exception):
    """Custom exception thrown when retry logic fails"""

    def __init__(self, retries_count, last_exception):
        message = 'Failed %s retries: %s' % (retries_count, last_exception)
        super(RetryError, self).__init__(message)
        self.last_exception = last_exception