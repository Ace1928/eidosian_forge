from .domreg import getDOMImplementation, registerDOMImplementation
class NotSupportedErr(DOMException):
    code = NOT_SUPPORTED_ERR