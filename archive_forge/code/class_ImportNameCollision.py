from .errors import BzrError, InternalBzrError
class ImportNameCollision(InternalBzrError):
    _fmt = 'Tried to import an object to the same name as an existing object. %(name)s'

    def __init__(self, name):
        BzrError.__init__(self)
        self.name = name