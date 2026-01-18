from functools import wraps
from importlib import import_module
from inspect import getfullargspec, unwrap
from django.utils.html import conditional_escape
from django.utils.itercompat import is_iterable
from .base import Node, Template, token_kwargs
from .exceptions import TemplateSyntaxError
class SimpleNode(TagHelperNode):
    child_nodelists = ()

    def __init__(self, func, takes_context, args, kwargs, target_var):
        super().__init__(func, takes_context, args, kwargs)
        self.target_var = target_var

    def render(self, context):
        resolved_args, resolved_kwargs = self.get_resolved_arguments(context)
        output = self.func(*resolved_args, **resolved_kwargs)
        if self.target_var is not None:
            context[self.target_var] = output
            return ''
        if context.autoescape:
            output = conditional_escape(output)
        return output