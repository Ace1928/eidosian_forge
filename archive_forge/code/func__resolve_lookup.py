import inspect
import logging
import re
from enum import Enum
from django.template.context import BaseContext
from django.utils.formats import localize
from django.utils.html import conditional_escape, escape
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import SafeData, SafeString, mark_safe
from django.utils.text import get_text_list, smart_split, unescape_string_literal
from django.utils.timezone import template_localtime
from django.utils.translation import gettext_lazy, pgettext_lazy
from .exceptions import TemplateSyntaxError
def _resolve_lookup(self, context):
    """
        Perform resolution of a real variable (i.e. not a literal) against the
        given context.

        As indicated by the method's name, this method is an implementation
        detail and shouldn't be called by external code. Use Variable.resolve()
        instead.
        """
    current = context
    try:
        for bit in self.lookups:
            try:
                current = current[bit]
            except (TypeError, AttributeError, KeyError, ValueError, IndexError):
                try:
                    if isinstance(current, BaseContext) and getattr(type(current), bit):
                        raise AttributeError
                    current = getattr(current, bit)
                except (TypeError, AttributeError):
                    if not isinstance(current, BaseContext) and bit in dir(current):
                        raise
                    try:
                        current = current[int(bit)]
                    except (IndexError, ValueError, KeyError, TypeError):
                        raise VariableDoesNotExist('Failed lookup for key [%s] in %r', (bit, current))
            if callable(current):
                if getattr(current, 'do_not_call_in_templates', False):
                    pass
                elif getattr(current, 'alters_data', False):
                    current = context.template.engine.string_if_invalid
                else:
                    try:
                        current = current()
                    except TypeError:
                        try:
                            signature = inspect.signature(current)
                        except ValueError:
                            current = context.template.engine.string_if_invalid
                        else:
                            try:
                                signature.bind()
                            except TypeError:
                                current = context.template.engine.string_if_invalid
                            else:
                                raise
    except Exception as e:
        template_name = getattr(context, 'template_name', None) or 'unknown'
        logger.debug("Exception while resolving variable '%s' in template '%s'.", bit, template_name, exc_info=True)
        if getattr(e, 'silent_variable_failure', False):
            current = context.template.engine.string_if_invalid
        else:
            raise
    return current