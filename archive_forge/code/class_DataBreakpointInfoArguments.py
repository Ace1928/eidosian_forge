from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class DataBreakpointInfoArguments(BaseSchema):
    """
    Arguments for 'dataBreakpointInfo' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'variablesReference': {'type': 'integer', 'description': 'Reference to the Variable container if the data breakpoint is requested for a child of the container.'}, 'name': {'type': 'string', 'description': "The name of the Variable's child to obtain data breakpoint information for.\nIf variablesReference isn't provided, this can be an expression."}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, name, variablesReference=None, update_ids_from_dap=False, **kwargs):
        """
        :param string name: The name of the Variable's child to obtain data breakpoint information for.
        If variablesReference isn't provided, this can be an expression.
        :param integer variablesReference: Reference to the Variable container if the data breakpoint is requested for a child of the container.
        """
        self.name = name
        self.variablesReference = variablesReference
        if update_ids_from_dap:
            self.variablesReference = self._translate_id_from_dap(self.variablesReference)
        self.kwargs = kwargs

    @classmethod
    def update_dict_ids_from_dap(cls, dct):
        if 'variablesReference' in dct:
            dct['variablesReference'] = cls._translate_id_from_dap(dct['variablesReference'])
        return dct

    def to_dict(self, update_ids_to_dap=False):
        name = self.name
        variablesReference = self.variablesReference
        if update_ids_to_dap:
            if variablesReference is not None:
                variablesReference = self._translate_id_to_dap(variablesReference)
        dct = {'name': name}
        if variablesReference is not None:
            dct['variablesReference'] = variablesReference
        dct.update(self.kwargs)
        return dct

    @classmethod
    def update_dict_ids_to_dap(cls, dct):
        if 'variablesReference' in dct:
            dct['variablesReference'] = cls._translate_id_to_dap(dct['variablesReference'])
        return dct