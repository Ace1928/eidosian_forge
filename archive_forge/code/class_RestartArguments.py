from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class RestartArguments(BaseSchema):
    """
    Arguments for 'restart' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'arguments': {'oneOf': [{'$ref': '#/definitions/LaunchRequestArguments'}, {'$ref': '#/definitions/AttachRequestArguments'}], 'description': "The latest version of the 'launch' or 'attach' configuration."}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, arguments=None, update_ids_from_dap=False, **kwargs):
        """
        :param TypeNA arguments: The latest version of the 'launch' or 'attach' configuration.
        """
        self.arguments = arguments
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        arguments = self.arguments
        dct = {}
        if arguments is not None:
            dct['arguments'] = arguments
        dct.update(self.kwargs)
        return dct