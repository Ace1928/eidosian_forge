from .domreg import getDOMImplementation, registerDOMImplementation
class WrongDocumentErr(DOMException):
    code = WRONG_DOCUMENT_ERR