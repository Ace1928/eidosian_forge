import hashlib
from functools import cached_property
from inspect import isawaitable
from sentry_sdk import configure_scope, start_span
from sentry_sdk.consts import OP
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
class SentryAsyncExtension(SchemaExtension):

    def __init__(self, *, execution_context=None):
        if execution_context:
            self.execution_context = execution_context

    @cached_property
    def _resource_name(self):
        query_hash = self.hash_query(self.execution_context.query)
        if self.execution_context.operation_name:
            return '{}:{}'.format(self.execution_context.operation_name, query_hash)
        return query_hash

    def hash_query(self, query):
        return hashlib.md5(query.encode('utf-8')).hexdigest()

    def on_operation(self):
        self._operation_name = self.execution_context.operation_name
        operation_type = 'query'
        op = OP.GRAPHQL_QUERY
        if self.execution_context.query.strip().startswith('mutation'):
            operation_type = 'mutation'
            op = OP.GRAPHQL_MUTATION
        elif self.execution_context.query.strip().startswith('subscription'):
            operation_type = 'subscription'
            op = OP.GRAPHQL_SUBSCRIPTION
        description = operation_type
        if self._operation_name:
            description += ' {}'.format(self._operation_name)
        Hub.current.add_breadcrumb(category='graphql.operation', data={'operation_name': self._operation_name, 'operation_type': operation_type})
        with configure_scope() as scope:
            if scope.span:
                self.graphql_span = scope.span.start_child(op=op, description=description)
            else:
                self.graphql_span = start_span(op=op, description=description)
        self.graphql_span.set_data('graphql.operation.type', operation_type)
        self.graphql_span.set_data('graphql.operation.name', self._operation_name)
        self.graphql_span.set_data('graphql.document', self.execution_context.query)
        self.graphql_span.set_data('graphql.resource_name', self._resource_name)
        yield
        self.graphql_span.finish()

    def on_validate(self):
        self.validation_span = self.graphql_span.start_child(op=OP.GRAPHQL_VALIDATE, description='validation')
        yield
        self.validation_span.finish()

    def on_parse(self):
        self.parsing_span = self.graphql_span.start_child(op=OP.GRAPHQL_PARSE, description='parsing')
        yield
        self.parsing_span.finish()

    def should_skip_tracing(self, _next, info):
        return strawberry_should_skip_tracing(_next, info)

    async def _resolve(self, _next, root, info, *args, **kwargs):
        result = _next(root, info, *args, **kwargs)
        if isawaitable(result):
            result = await result
        return result

    async def resolve(self, _next, root, info, *args, **kwargs):
        if self.should_skip_tracing(_next, info):
            return await self._resolve(_next, root, info, *args, **kwargs)
        field_path = '{}.{}'.format(info.parent_type, info.field_name)
        with self.graphql_span.start_child(op=OP.GRAPHQL_RESOLVE, description='resolving {}'.format(field_path)) as span:
            span.set_data('graphql.field_name', info.field_name)
            span.set_data('graphql.parent_type', info.parent_type.name)
            span.set_data('graphql.field_path', field_path)
            span.set_data('graphql.path', '.'.join(map(str, info.path.as_list())))
            return await self._resolve(_next, root, info, *args, **kwargs)