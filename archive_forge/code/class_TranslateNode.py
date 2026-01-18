from decimal import Decimal
from django.conf import settings
from django.template import Library, Node, TemplateSyntaxError, Variable
from django.template.base import TokenType, render_value_in_context
from django.template.defaulttags import token_kwargs
from django.utils import translation
from django.utils.safestring import SafeData, SafeString, mark_safe
class TranslateNode(Node):
    child_nodelists = ()

    def __init__(self, filter_expression, noop, asvar=None, message_context=None):
        self.noop = noop
        self.asvar = asvar
        self.message_context = message_context
        self.filter_expression = filter_expression
        if isinstance(self.filter_expression.var, str):
            self.filter_expression.is_var = True
            self.filter_expression.var = Variable("'%s'" % self.filter_expression.var)

    def render(self, context):
        self.filter_expression.var.translate = not self.noop
        if self.message_context:
            self.filter_expression.var.message_context = self.message_context.resolve(context)
        output = self.filter_expression.resolve(context)
        value = render_value_in_context(output, context)
        is_safe = isinstance(value, SafeData)
        value = value.replace('%%', '%')
        value = mark_safe(value) if is_safe else value
        if self.asvar:
            context[self.asvar] = value
            return ''
        else:
            return value