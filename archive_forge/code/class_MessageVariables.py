from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class MessageVariables(BaseSchema):
    """
    "variables" of Message

    Note: automatically generated code. Do not edit manually.
    """
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