from __future__ import absolute_import
import re
from sentry_sdk._types import TYPE_CHECKING
from django import VERSION as DJANGO_VERSION
def _resolve(self, resolver, path, parents=None):
    match = get_regex(resolver).search(path)
    if not match:
        return None
    if parents is None:
        parents = [resolver]
    elif resolver not in parents:
        parents = parents + [resolver]
    new_path = path[match.end():]
    for pattern in resolver.url_patterns:
        if not pattern.callback:
            match_ = self._resolve(pattern, new_path, parents)
            if match_:
                return match_
            continue
        elif not get_regex(pattern).search(new_path):
            continue
        try:
            return self._cache[pattern]
        except KeyError:
            pass
        prefix = ''.join((self._simplify(p) for p in parents))
        result = prefix + self._simplify(pattern)
        if not result.startswith('/'):
            result = '/' + result
        self._cache[pattern] = result
        return result
    return None