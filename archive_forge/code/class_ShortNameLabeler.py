import re
from pyomo.common.deprecation import deprecated
from pyomo.core.base.componentuid import ComponentUID
class ShortNameLabeler(object):

    def __init__(self, limit, suffix, start=0, labeler=None, prefix='', caseInsensitive=False, legalRegex=None):
        self.id = start
        self.prefix = prefix
        self.suffix = suffix
        self.limit = limit
        if labeler is not None:
            self.labeler = labeler
        else:
            self.labeler = AlphaNumericTextLabeler()
        self.known_labels = set()
        self.caseInsensitive = caseInsensitive
        if isinstance(legalRegex, str):
            self.legalRegex = re.compile(legalRegex)
        else:
            self.legalRegex = legalRegex

    def __call__(self, obj=None):
        lbl = self.labeler(obj)
        lbl_len = len(lbl)
        shorten = False
        if lbl_len > self.limit:
            shorten = True
        elif lbl_len == self.limit and lbl.startswith(self.prefix) and lbl.endswith(self.suffix):
            shorten = True
        elif (lbl.upper() if self.caseInsensitive else lbl) in self.known_labels:
            shorten = True
        elif self.legalRegex and (not self.legalRegex.match(lbl)):
            shorten = True
        if shorten:
            self.id += 1
            suffix = '%s%d%s' % (self.suffix, self.id, self.suffix)
            tail = -self.limit + len(suffix) + len(self.prefix)
            if tail >= 0:
                raise RuntimeError('Too many identifiers.\n\tThe ShortNameLabeler cannot generate a guaranteed unique label limited to %d characters' % (self.limit,))
            lbl = self.prefix + lbl[tail:] + suffix
        if self.known_labels is not None:
            self.known_labels.add(lbl.upper() if self.caseInsensitive else lbl)
        return lbl