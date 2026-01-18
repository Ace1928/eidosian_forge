from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class SetDebuggerPropertyArguments(BaseSchema):
    """
    Arguments for 'setDebuggerProperty' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'ideOS': {'type': ['string'], 'description': 'OS where the ide is running. Supported values [Windows, Linux]'}, 'dontTraceStartPatterns': {'type': ['array'], 'description': 'Patterns to match with the start of the file paths. Matching paths will be added to a list of file where trace is ignored.'}, 'dontTraceEndPatterns': {'type': ['array'], 'description': 'Patterns to match with the end of the file paths. Matching paths will be added to a list of file where trace is ignored.'}, 'skipSuspendOnBreakpointException': {'type': ['array'], 'description': 'List of exceptions that should be skipped when doing condition evaluations.'}, 'skipPrintBreakpointException': {'type': ['array'], 'description': 'List of exceptions that should skip printing to stderr when doing condition evaluations.'}, 'multiThreadsSingleNotification': {'type': ['boolean'], 'description': 'If false then a notification is generated for each thread event. If true a single event is gnenerated, and all threads follow that behavior.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, ideOS=None, dontTraceStartPatterns=None, dontTraceEndPatterns=None, skipSuspendOnBreakpointException=None, skipPrintBreakpointException=None, multiThreadsSingleNotification=None, update_ids_from_dap=False, **kwargs):
        """
        :param ['string'] ideOS: OS where the ide is running. Supported values [Windows, Linux]
        :param ['array'] dontTraceStartPatterns: Patterns to match with the start of the file paths. Matching paths will be added to a list of file where trace is ignored.
        :param ['array'] dontTraceEndPatterns: Patterns to match with the end of the file paths. Matching paths will be added to a list of file where trace is ignored.
        :param ['array'] skipSuspendOnBreakpointException: List of exceptions that should be skipped when doing condition evaluations.
        :param ['array'] skipPrintBreakpointException: List of exceptions that should skip printing to stderr when doing condition evaluations.
        :param ['boolean'] multiThreadsSingleNotification: If false then a notification is generated for each thread event. If true a single event is gnenerated, and all threads follow that behavior.
        """
        self.ideOS = ideOS
        self.dontTraceStartPatterns = dontTraceStartPatterns
        self.dontTraceEndPatterns = dontTraceEndPatterns
        self.skipSuspendOnBreakpointException = skipSuspendOnBreakpointException
        self.skipPrintBreakpointException = skipPrintBreakpointException
        self.multiThreadsSingleNotification = multiThreadsSingleNotification
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        ideOS = self.ideOS
        dontTraceStartPatterns = self.dontTraceStartPatterns
        dontTraceEndPatterns = self.dontTraceEndPatterns
        skipSuspendOnBreakpointException = self.skipSuspendOnBreakpointException
        skipPrintBreakpointException = self.skipPrintBreakpointException
        multiThreadsSingleNotification = self.multiThreadsSingleNotification
        dct = {}
        if ideOS is not None:
            dct['ideOS'] = ideOS
        if dontTraceStartPatterns is not None:
            dct['dontTraceStartPatterns'] = dontTraceStartPatterns
        if dontTraceEndPatterns is not None:
            dct['dontTraceEndPatterns'] = dontTraceEndPatterns
        if skipSuspendOnBreakpointException is not None:
            dct['skipSuspendOnBreakpointException'] = skipSuspendOnBreakpointException
        if skipPrintBreakpointException is not None:
            dct['skipPrintBreakpointException'] = skipPrintBreakpointException
        if multiThreadsSingleNotification is not None:
            dct['multiThreadsSingleNotification'] = multiThreadsSingleNotification
        dct.update(self.kwargs)
        return dct