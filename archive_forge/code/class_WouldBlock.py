from trio._util import NoPublicConstructor, final
class WouldBlock(Exception):
    """Raised by ``X_nowait`` functions if ``X`` would block."""