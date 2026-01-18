from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ErrorResponseBody(BaseSchema):
    """
    "body" of ErrorResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'error': {'description': 'An optional, structured error message.', 'type': 'Message'}}
    __refs__ = set(['error'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, error=None, update_ids_from_dap=False, **kwargs):
        """
        :param Message error: An optional, structured error message.
        """
        if error is None:
            self.error = Message()
        else:
            self.error = Message(update_ids_from_dap=update_ids_from_dap, **error) if error.__class__ != Message else error
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        error = self.error
        dct = {}
        if error is not None:
            dct['error'] = error.to_dict(update_ids_to_dap=update_ids_to_dap)
        dct.update(self.kwargs)
        return dct