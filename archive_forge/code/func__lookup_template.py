import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def _lookup_template(context, uri, relativeto):
    lookup = context._with_template.lookup
    if lookup is None:
        raise exceptions.TemplateLookupException("Template '%s' has no TemplateLookup associated" % context._with_template.uri)
    uri = lookup.adjust_uri(uri, relativeto)
    try:
        return lookup.get_template(uri)
    except exceptions.TopLevelLookupException as e:
        raise exceptions.TemplateLookupException(str(compat.exception_as())) from e