from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class SetVariableArguments(BaseSchema):
    """
    Arguments for 'setVariable' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'variablesReference': {'type': 'integer', 'description': 'The reference of the variable container.'}, 'name': {'type': 'string', 'description': 'The name of the variable in the container.'}, 'value': {'type': 'string', 'description': 'The value of the variable.'}, 'format': {'description': 'Specifies details on how to format the response value.', 'type': 'ValueFormat'}}
    __refs__ = set(['format'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, variablesReference, name, value, format=None, update_ids_from_dap=False, **kwargs):
        """
        :param integer variablesReference: The reference of the variable container.
        :param string name: The name of the variable in the container.
        :param string value: The value of the variable.
        :param ValueFormat format: Specifies details on how to format the response value.
        """
        self.variablesReference = variablesReference
        self.name = name
        self.value = value
        if format is None:
            self.format = ValueFormat()
        else:
            self.format = ValueFormat(update_ids_from_dap=update_ids_from_dap, **format) if format.__class__ != ValueFormat else format
        if update_ids_from_dap:
            self.variablesReference = self._translate_id_from_dap(self.variablesReference)
        self.kwargs = kwargs

    @classmethod
    def update_dict_ids_from_dap(cls, dct):
        if 'variablesReference' in dct:
            dct['variablesReference'] = cls._translate_id_from_dap(dct['variablesReference'])
        return dct

    def to_dict(self, update_ids_to_dap=False):
        variablesReference = self.variablesReference
        name = self.name
        value = self.value
        format = self.format
        if update_ids_to_dap:
            if variablesReference is not None:
                variablesReference = self._translate_id_to_dap(variablesReference)
        dct = {'variablesReference': variablesReference, 'name': name, 'value': value}
        if format is not None:
            dct['format'] = format.to_dict(update_ids_to_dap=update_ids_to_dap)
        dct.update(self.kwargs)
        return dct

    @classmethod
    def update_dict_ids_to_dap(cls, dct):
        if 'variablesReference' in dct:
            dct['variablesReference'] = cls._translate_id_to_dap(dct['variablesReference'])
        return dct