from types import NoneType
from django.db.backends.utils import names_digest, split_identifier
from django.db.models.expressions import Col, ExpressionList, F, Func, OrderBy
from django.db.models.functions import Collate
from django.db.models.query_utils import Q
from django.db.models.sql import Query
from django.utils.functional import partition
class IndexExpression(Func):
    """Order and wrap expressions for CREATE INDEX statements."""
    template = '%(expressions)s'
    wrapper_classes = (OrderBy, Collate)

    def set_wrapper_classes(self, connection=None):
        if connection and connection.features.collate_as_index_expression:
            self.wrapper_classes = tuple([wrapper_cls for wrapper_cls in self.wrapper_classes if wrapper_cls is not Collate])

    @classmethod
    def register_wrappers(cls, *wrapper_classes):
        cls.wrapper_classes = wrapper_classes

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        expressions = list(self.flatten())
        index_expressions, wrappers = partition(lambda e: isinstance(e, self.wrapper_classes), expressions)
        wrapper_types = [type(wrapper) for wrapper in wrappers]
        if len(wrapper_types) != len(set(wrapper_types)):
            raise ValueError("Multiple references to %s can't be used in an indexed expression." % ', '.join([wrapper_cls.__qualname__ for wrapper_cls in self.wrapper_classes]))
        if expressions[1:len(wrappers) + 1] != wrappers:
            raise ValueError('%s must be topmost expressions in an indexed expression.' % ', '.join([wrapper_cls.__qualname__ for wrapper_cls in self.wrapper_classes]))
        root_expression = index_expressions[1]
        resolve_root_expression = root_expression.resolve_expression(query, allow_joins, reuse, summarize, for_save)
        if not isinstance(resolve_root_expression, Col):
            root_expression = Func(root_expression, template='(%(expressions)s)')
        if wrappers:
            wrappers = sorted(wrappers, key=lambda w: self.wrapper_classes.index(type(w)))
            wrappers = [wrapper.copy() for wrapper in wrappers]
            for i, wrapper in enumerate(wrappers[:-1]):
                wrapper.set_source_expressions([wrappers[i + 1]])
            wrappers[-1].set_source_expressions([root_expression])
            self.set_source_expressions([wrappers[0]])
        else:
            self.set_source_expressions([root_expression])
        return super().resolve_expression(query, allow_joins, reuse, summarize, for_save)

    def as_sqlite(self, compiler, connection, **extra_context):
        return self.as_sql(compiler, connection, **extra_context)