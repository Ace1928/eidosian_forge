from .domreg import getDOMImplementation, registerDOMImplementation
class InvalidStateErr(DOMException):
    code = INVALID_STATE_ERR