from .domreg import getDOMImplementation, registerDOMImplementation
class InvalidCharacterErr(DOMException):
    code = INVALID_CHARACTER_ERR