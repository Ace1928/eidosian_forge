from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class CompletionItemType(BaseSchema):
    """
    Some predefined types for the CompletionItem. Please note that not all clients have specific icons
    for all of them.

    Note: automatically generated code. Do not edit manually.
    """
    METHOD = 'method'
    FUNCTION = 'function'
    CONSTRUCTOR = 'constructor'
    FIELD = 'field'
    VARIABLE = 'variable'
    CLASS = 'class'
    INTERFACE = 'interface'
    MODULE = 'module'
    PROPERTY = 'property'
    UNIT = 'unit'
    VALUE = 'value'
    ENUM = 'enum'
    KEYWORD = 'keyword'
    SNIPPET = 'snippet'
    TEXT = 'text'
    COLOR = 'color'
    FILE = 'file'
    REFERENCE = 'reference'
    CUSTOMCOLOR = 'customcolor'
    VALID_VALUES = set(['method', 'function', 'constructor', 'field', 'variable', 'class', 'interface', 'module', 'property', 'unit', 'value', 'enum', 'keyword', 'snippet', 'text', 'color', 'file', 'reference', 'customcolor'])
    __props__ = {}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, update_ids_from_dap=False, **kwargs):
        """
    
        """
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        dct = {}
        dct.update(self.kwargs)
        return dct