class UnsupportedFormatError(BzrError):
    _fmt = "Unsupported branch format: %(format)s\nPlease run 'brz upgrade'"