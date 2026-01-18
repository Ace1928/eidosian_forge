class BidiConnection:

    def __init__(self, session, cdp, devtools_import) -> None:
        self.session = session
        self.cdp = cdp
        self.devtools = devtools_import