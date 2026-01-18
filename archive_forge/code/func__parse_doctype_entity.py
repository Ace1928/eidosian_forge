import re
def _parse_doctype_entity(self, i, declstartpos):
    rawdata = self.rawdata
    if rawdata[i:i + 1] == '%':
        j = i + 1
        while 1:
            c = rawdata[j:j + 1]
            if not c:
                return -1
            if c.isspace():
                j = j + 1
            else:
                break
    else:
        j = i
    name, j = self._scan_name(j, declstartpos)
    if j < 0:
        return j
    while 1:
        c = self.rawdata[j:j + 1]
        if not c:
            return -1
        if c in '\'"':
            m = _declstringlit_match(rawdata, j)
            if m:
                j = m.end()
            else:
                return -1
        elif c == '>':
            return j + 1
        else:
            name, j = self._scan_name(j, declstartpos)
            if j < 0:
                return j