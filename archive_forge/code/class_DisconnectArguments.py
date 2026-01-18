from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class DisconnectArguments(BaseSchema):
    """
    Arguments for 'disconnect' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'restart': {'type': 'boolean', 'description': "A value of true indicates that this 'disconnect' request is part of a restart sequence."}, 'terminateDebuggee': {'type': 'boolean', 'description': "Indicates whether the debuggee should be terminated when the debugger is disconnected.\nIf unspecified, the debug adapter is free to do whatever it thinks is best.\nThe attribute is only honored by a debug adapter if the capability 'supportTerminateDebuggee' is true."}, 'suspendDebuggee': {'type': 'boolean', 'description': "Indicates whether the debuggee should stay suspended when the debugger is disconnected.\nIf unspecified, the debuggee should resume execution.\nThe attribute is only honored by a debug adapter if the capability 'supportSuspendDebuggee' is true."}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, restart=None, terminateDebuggee=None, suspendDebuggee=None, update_ids_from_dap=False, **kwargs):
        """
        :param boolean restart: A value of true indicates that this 'disconnect' request is part of a restart sequence.
        :param boolean terminateDebuggee: Indicates whether the debuggee should be terminated when the debugger is disconnected.
        If unspecified, the debug adapter is free to do whatever it thinks is best.
        The attribute is only honored by a debug adapter if the capability 'supportTerminateDebuggee' is true.
        :param boolean suspendDebuggee: Indicates whether the debuggee should stay suspended when the debugger is disconnected.
        If unspecified, the debuggee should resume execution.
        The attribute is only honored by a debug adapter if the capability 'supportSuspendDebuggee' is true.
        """
        self.restart = restart
        self.terminateDebuggee = terminateDebuggee
        self.suspendDebuggee = suspendDebuggee
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        restart = self.restart
        terminateDebuggee = self.terminateDebuggee
        suspendDebuggee = self.suspendDebuggee
        dct = {}
        if restart is not None:
            dct['restart'] = restart
        if terminateDebuggee is not None:
            dct['terminateDebuggee'] = terminateDebuggee
        if suspendDebuggee is not None:
            dct['suspendDebuggee'] = suspendDebuggee
        dct.update(self.kwargs)
        return dct