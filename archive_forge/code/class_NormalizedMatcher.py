import logging
import re
from .compat import string_types
from .util import parse_requirement
class NormalizedMatcher(Matcher):
    version_class = NormalizedVersion
    _operators = {'~=': '_match_compatible', '<': '_match_lt', '>': '_match_gt', '<=': '_match_le', '>=': '_match_ge', '==': '_match_eq', '===': '_match_arbitrary', '!=': '_match_ne'}

    def _adjust_local(self, version, constraint, prefix):
        if prefix:
            strip_local = '+' not in constraint and version._parts[-1]
        else:
            strip_local = not constraint._parts[-1] and version._parts[-1]
        if strip_local:
            s = version._string.split('+', 1)[0]
            version = self.version_class(s)
        return (version, constraint)

    def _match_lt(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        if version >= constraint:
            return False
        release_clause = constraint._release_clause
        pfx = '.'.join([str(i) for i in release_clause])
        return not _match_prefix(version, pfx)

    def _match_gt(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        if version <= constraint:
            return False
        release_clause = constraint._release_clause
        pfx = '.'.join([str(i) for i in release_clause])
        return not _match_prefix(version, pfx)

    def _match_le(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        return version <= constraint

    def _match_ge(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        return version >= constraint

    def _match_eq(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        if not prefix:
            result = version == constraint
        else:
            result = _match_prefix(version, constraint)
        return result

    def _match_arbitrary(self, version, constraint, prefix):
        return str(version) == str(constraint)

    def _match_ne(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        if not prefix:
            result = version != constraint
        else:
            result = not _match_prefix(version, constraint)
        return result

    def _match_compatible(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        if version == constraint:
            return True
        if version < constraint:
            return False
        release_clause = constraint._release_clause
        if len(release_clause) > 1:
            release_clause = release_clause[:-1]
        pfx = '.'.join([str(i) for i in release_clause])
        return _match_prefix(version, pfx)