from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ExceptionInfoResponseBody(BaseSchema):
    """
    "body" of ExceptionInfoResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'exceptionId': {'type': 'string', 'description': 'ID of the exception that was thrown.'}, 'description': {'type': 'string', 'description': 'Descriptive text for the exception provided by the debug adapter.'}, 'breakMode': {'description': 'Mode that caused the exception notification to be raised.', 'type': 'ExceptionBreakMode'}, 'details': {'description': 'Detailed information about the exception.', 'type': 'ExceptionDetails'}}
    __refs__ = set(['breakMode', 'details'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, exceptionId, breakMode, description=None, details=None, update_ids_from_dap=False, **kwargs):
        """
        :param string exceptionId: ID of the exception that was thrown.
        :param ExceptionBreakMode breakMode: Mode that caused the exception notification to be raised.
        :param string description: Descriptive text for the exception provided by the debug adapter.
        :param ExceptionDetails details: Detailed information about the exception.
        """
        self.exceptionId = exceptionId
        if breakMode is not None:
            assert breakMode in ExceptionBreakMode.VALID_VALUES
        self.breakMode = breakMode
        self.description = description
        if details is None:
            self.details = ExceptionDetails()
        else:
            self.details = ExceptionDetails(update_ids_from_dap=update_ids_from_dap, **details) if details.__class__ != ExceptionDetails else details
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        exceptionId = self.exceptionId
        breakMode = self.breakMode
        description = self.description
        details = self.details
        dct = {'exceptionId': exceptionId, 'breakMode': breakMode}
        if description is not None:
            dct['description'] = description
        if details is not None:
            dct['details'] = details.to_dict(update_ids_to_dap=update_ids_to_dap)
        dct.update(self.kwargs)
        return dct