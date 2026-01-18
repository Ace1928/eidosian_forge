import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def _find_all(self, name, attrs, string, limit, generator, **kwargs):
    """Iterates over a generator looking for things that match."""
    _stacklevel = kwargs.pop('_stacklevel', 3)
    if string is None and 'text' in kwargs:
        string = kwargs.pop('text')
        warnings.warn("The 'text' argument to find()-type methods is deprecated. Use 'string' instead.", DeprecationWarning, stacklevel=_stacklevel)
    if isinstance(name, SoupStrainer):
        strainer = name
    else:
        strainer = SoupStrainer(name, attrs, string, **kwargs)
    if string is None and (not limit) and (not attrs) and (not kwargs):
        if name is True or name is None:
            result = (element for element in generator if isinstance(element, Tag))
            return ResultSet(strainer, result)
        elif isinstance(name, str):
            if name.count(':') == 1:
                prefix, local_name = name.split(':', 1)
            else:
                prefix = None
                local_name = name
            result = (element for element in generator if isinstance(element, Tag) and element.name == name or (element.name == local_name and (prefix is None or element.prefix == prefix)))
            return ResultSet(strainer, result)
    results = ResultSet(strainer)
    while True:
        try:
            i = next(generator)
        except StopIteration:
            break
        if i:
            found = strainer.search(i)
            if found:
                results.append(found)
                if limit and len(results) >= limit:
                    break
    return results