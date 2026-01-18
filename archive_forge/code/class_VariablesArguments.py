from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class VariablesArguments(BaseSchema):
    """
    Arguments for 'variables' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'variablesReference': {'type': 'integer', 'description': 'The Variable reference.'}, 'filter': {'type': 'string', 'enum': ['indexed', 'named'], 'description': 'Optional filter to limit the child variables to either named or indexed. If omitted, both types are fetched.'}, 'start': {'type': 'integer', 'description': 'The index of the first variable to return; if omitted children start at 0.'}, 'count': {'type': 'integer', 'description': 'The number of variables to return. If count is missing or 0, all variables are returned.'}, 'format': {'description': "Specifies details on how to format the Variable values.\nThe attribute is only honored by a debug adapter if the capability 'supportsValueFormattingOptions' is true.", 'type': 'ValueFormat'}}
    __refs__ = set(['format'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, variablesReference, filter=None, start=None, count=None, format=None, update_ids_from_dap=False, **kwargs):
        """
        :param integer variablesReference: The Variable reference.
        :param string filter: Optional filter to limit the child variables to either named or indexed. If omitted, both types are fetched.
        :param integer start: The index of the first variable to return; if omitted children start at 0.
        :param integer count: The number of variables to return. If count is missing or 0, all variables are returned.
        :param ValueFormat format: Specifies details on how to format the Variable values.
        The attribute is only honored by a debug adapter if the capability 'supportsValueFormattingOptions' is true.
        """
        self.variablesReference = variablesReference
        self.filter = filter
        self.start = start
        self.count = count
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
        filter = self.filter
        start = self.start
        count = self.count
        format = self.format
        if update_ids_to_dap:
            if variablesReference is not None:
                variablesReference = self._translate_id_to_dap(variablesReference)
        dct = {'variablesReference': variablesReference}
        if filter is not None:
            dct['filter'] = filter
        if start is not None:
            dct['start'] = start
        if count is not None:
            dct['count'] = count
        if format is not None:
            dct['format'] = format.to_dict(update_ids_to_dap=update_ids_to_dap)
        dct.update(self.kwargs)
        return dct

    @classmethod
    def update_dict_ids_to_dap(cls, dct):
        if 'variablesReference' in dct:
            dct['variablesReference'] = cls._translate_id_to_dap(dct['variablesReference'])
        return dct