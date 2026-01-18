import logging
from wandb_graphql import parse, introspection_query, build_ast_schema, build_client_schema
from wandb_graphql.validation import validate
from .transport.local_schema import LocalSchemaTransport
def _get_result(self, document, *args, **kwargs):
    if not self.retries:
        return self.transport.execute(document, *args, **kwargs)
    last_exception = None
    retries_count = 0
    while retries_count < self.retries:
        try:
            result = self.transport.execute(document, *args, **kwargs)
            return result
        except Exception as e:
            last_exception = e
            log.warning('Request failed with exception %s. Retrying for the %s time...', e, retries_count + 1, exc_info=True)
        finally:
            retries_count += 1
    raise RetryError(retries_count, last_exception)