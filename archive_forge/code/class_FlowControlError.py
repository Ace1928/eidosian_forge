import h2.errors
class FlowControlError(ProtocolError):
    """
    An attempted action violates flow control constraints.
    """
    error_code = h2.errors.ErrorCodes.FLOW_CONTROL_ERROR