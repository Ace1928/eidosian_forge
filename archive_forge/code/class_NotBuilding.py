from . import errors
class NotBuilding(errors.BzrError):
    _fmt = 'Not currently building a tree.'