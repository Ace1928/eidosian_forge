from django.db.models import (
from django.db.models.expressions import CombinedExpression, register_combinable_fields
from django.db.models.functions import Cast, Coalesce
class SearchVector(SearchVectorCombinable, Func):
    function = 'to_tsvector'
    arg_joiner = " || ' ' || "
    output_field = SearchVectorField()

    def __init__(self, *expressions, config=None, weight=None):
        super().__init__(*expressions)
        self.config = SearchConfig.from_parameter(config)
        if weight is not None and (not hasattr(weight, 'resolve_expression')):
            weight = Value(weight)
        self.weight = weight

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        resolved = super().resolve_expression(query, allow_joins, reuse, summarize, for_save)
        if self.config:
            resolved.config = self.config.resolve_expression(query, allow_joins, reuse, summarize, for_save)
        return resolved

    def as_sql(self, compiler, connection, function=None, template=None):
        clone = self.copy()
        clone.set_source_expressions([Coalesce(expression if isinstance(expression.output_field, (CharField, TextField)) else Cast(expression, TextField()), Value('')) for expression in clone.get_source_expressions()])
        config_sql = None
        config_params = []
        if template is None:
            if clone.config:
                config_sql, config_params = compiler.compile(clone.config)
                template = '%(function)s(%(config)s, %(expressions)s)'
            else:
                template = clone.template
        sql, params = super(SearchVector, clone).as_sql(compiler, connection, function=function, template=template, config=config_sql)
        extra_params = []
        if clone.weight:
            weight_sql, extra_params = compiler.compile(clone.weight)
            sql = 'setweight({}, {})'.format(sql, weight_sql)
        return (sql, config_params + params + extra_params)