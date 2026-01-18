class Scheme:
    """A Scheme holds paths which are used as the base directories for
    artifacts associated with a Python package.
    """
    __slots__ = SCHEME_KEYS

    def __init__(self, platlib: str, purelib: str, headers: str, scripts: str, data: str) -> None:
        self.platlib = platlib
        self.purelib = purelib
        self.headers = headers
        self.scripts = scripts
        self.data = data