import re
def _parse_doctype_attlist(self, i, declstartpos):
    rawdata = self.rawdata
    name, j = self._scan_name(i, declstartpos)
    c = rawdata[j:j + 1]
    if c == '':
        return -1
    if c == '>':
        return j + 1
    while 1:
        name, j = self._scan_name(j, declstartpos)
        if j < 0:
            return j
        c = rawdata[j:j + 1]
        if c == '':
            return -1
        if c == '(':
            if ')' in rawdata[j:]:
                j = rawdata.find(')', j) + 1
            else:
                return -1
            while rawdata[j:j + 1].isspace():
                j = j + 1
            if not rawdata[j:]:
                return -1
        else:
            name, j = self._scan_name(j, declstartpos)
        c = rawdata[j:j + 1]
        if not c:
            return -1
        if c in '\'"':
            m = _declstringlit_match(rawdata, j)
            if m:
                j = m.end()
            else:
                return -1
            c = rawdata[j:j + 1]
            if not c:
                return -1
        if c == '#':
            if rawdata[j:] == '#':
                return -1
            name, j = self._scan_name(j + 1, declstartpos)
            if j < 0:
                return j
            c = rawdata[j:j + 1]
            if not c:
                return -1
        if c == '>':
            return j + 1