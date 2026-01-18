from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class StackFrameFormat(BaseSchema):
    """
    Provides formatting information for a stack frame.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'hex': {'type': 'boolean', 'description': 'Display the value in hex.'}, 'parameters': {'type': 'boolean', 'description': 'Displays parameters for the stack frame.'}, 'parameterTypes': {'type': 'boolean', 'description': 'Displays the types of parameters for the stack frame.'}, 'parameterNames': {'type': 'boolean', 'description': 'Displays the names of parameters for the stack frame.'}, 'parameterValues': {'type': 'boolean', 'description': 'Displays the values of parameters for the stack frame.'}, 'line': {'type': 'boolean', 'description': 'Displays the line number of the stack frame.'}, 'module': {'type': 'boolean', 'description': 'Displays the module of the stack frame.'}, 'includeAll': {'type': 'boolean', 'description': 'Includes all stack frames, including those the debug adapter might otherwise hide.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, hex=None, parameters=None, parameterTypes=None, parameterNames=None, parameterValues=None, line=None, module=None, includeAll=None, update_ids_from_dap=False, **kwargs):
        """
        :param boolean hex: Display the value in hex.
        :param boolean parameters: Displays parameters for the stack frame.
        :param boolean parameterTypes: Displays the types of parameters for the stack frame.
        :param boolean parameterNames: Displays the names of parameters for the stack frame.
        :param boolean parameterValues: Displays the values of parameters for the stack frame.
        :param boolean line: Displays the line number of the stack frame.
        :param boolean module: Displays the module of the stack frame.
        :param boolean includeAll: Includes all stack frames, including those the debug adapter might otherwise hide.
        """
        self.hex = hex
        self.parameters = parameters
        self.parameterTypes = parameterTypes
        self.parameterNames = parameterNames
        self.parameterValues = parameterValues
        self.line = line
        self.module = module
        self.includeAll = includeAll
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        hex = self.hex
        parameters = self.parameters
        parameterTypes = self.parameterTypes
        parameterNames = self.parameterNames
        parameterValues = self.parameterValues
        line = self.line
        module = self.module
        includeAll = self.includeAll
        dct = {}
        if hex is not None:
            dct['hex'] = hex
        if parameters is not None:
            dct['parameters'] = parameters
        if parameterTypes is not None:
            dct['parameterTypes'] = parameterTypes
        if parameterNames is not None:
            dct['parameterNames'] = parameterNames
        if parameterValues is not None:
            dct['parameterValues'] = parameterValues
        if line is not None:
            dct['line'] = line
        if module is not None:
            dct['module'] = module
        if includeAll is not None:
            dct['includeAll'] = includeAll
        dct.update(self.kwargs)
        return dct