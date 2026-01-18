class _UnicodeBuiltin(object):

    def __getitem__(self, charCode):
        try:
            import unicodedata2 as unicodedata
        except ImportError:
            import unicodedata
        try:
            return unicodedata.name(chr(charCode))
        except ValueError:
            return '????'