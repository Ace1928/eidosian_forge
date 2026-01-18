from .domreg import getDOMImplementation, registerDOMImplementation
class NoModificationAllowedErr(DOMException):
    code = NO_MODIFICATION_ALLOWED_ERR