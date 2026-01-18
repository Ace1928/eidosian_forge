from .. import errors, registry
class UnsupportedInventoryKind(errors.BzrError):
    _fmt = 'Unsupported entry kind %(kind)s'

    def __init__(self, kind):
        self.kind = kind