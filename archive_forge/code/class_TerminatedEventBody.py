from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class TerminatedEventBody(BaseSchema):
    """
    "body" of TerminatedEvent

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'restart': {'type': ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'], 'description': "A debug adapter may set 'restart' to true (or to an arbitrary object) to request that the front end restarts the session.\nThe value is not interpreted by the client and passed unmodified as an attribute '__restart' to the 'launch' and 'attach' requests."}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, restart=None, update_ids_from_dap=False, **kwargs):
        """
        :param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] restart: A debug adapter may set 'restart' to true (or to an arbitrary object) to request that the front end restarts the session.
        The value is not interpreted by the client and passed unmodified as an attribute '__restart' to the 'launch' and 'attach' requests.
        """
        self.restart = restart
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        restart = self.restart
        dct = {}
        if restart is not None:
            dct['restart'] = restart
        dct.update(self.kwargs)
        return dct