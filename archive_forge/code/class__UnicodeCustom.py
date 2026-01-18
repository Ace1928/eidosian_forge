class _UnicodeCustom(object):

    def __init__(self, f):
        if isinstance(f, str):
            with open(f) as fd:
                codes = _makeunicodes(fd)
        else:
            codes = _makeunicodes(f)
        self.codes = codes

    def __getitem__(self, charCode):
        try:
            return self.codes[charCode]
        except KeyError:
            return '????'