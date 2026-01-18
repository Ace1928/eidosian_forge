import re
def _scan_name(self, i, declstartpos):
    rawdata = self.rawdata
    n = len(rawdata)
    if i == n:
        return (None, -1)
    m = _declname_match(rawdata, i)
    if m:
        s = m.group()
        name = s.strip()
        if i + len(s) == n:
            return (None, -1)
        return (name.lower(), m.end())
    else:
        self.updatepos(declstartpos, i)
        raise AssertionError('expected name token at %r' % rawdata[declstartpos:declstartpos + 20])