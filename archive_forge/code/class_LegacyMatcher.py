import logging
import re
from .compat import string_types
from .util import parse_requirement
class LegacyMatcher(Matcher):
    version_class = LegacyVersion
    _operators = dict(Matcher._operators)
    _operators['~='] = '_match_compatible'
    numeric_re = re.compile('^(\\d+(\\.\\d+)*)')

    def _match_compatible(self, version, constraint, prefix):
        if version < constraint:
            return False
        m = self.numeric_re.match(str(constraint))
        if not m:
            logger.warning('Cannot compute compatible match for version %s  and constraint %s', version, constraint)
            return True
        s = m.groups()[0]
        if '.' in s:
            s = s.rsplit('.', 1)[0]
        return _match_prefix(version, s)