from .. import errors, registry
class BadInventoryFormat(errors.BzrError):
    _fmt = 'Root class for inventory serialization errors'