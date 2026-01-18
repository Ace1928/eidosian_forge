import re
from fqdn._compat import cached_property
@property
def _regex(self):
    regexstr = FQDN.PREFERRED_NAME_SYNTAX_REGEXSTR if not self._allow_underscores else FQDN.ALLOW_UNDERSCORES_REGEXSTR
    return re.compile(regexstr, re.IGNORECASE)