from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class CapabilitiesEventBody(BaseSchema):
    """
    "body" of CapabilitiesEvent

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'capabilities': {'description': 'The set of updated capabilities.', 'type': 'Capabilities'}}
    __refs__ = set(['capabilities'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, capabilities, update_ids_from_dap=False, **kwargs):
        """
        :param Capabilities capabilities: The set of updated capabilities.
        """
        if capabilities is None:
            self.capabilities = Capabilities()
        else:
            self.capabilities = Capabilities(update_ids_from_dap=update_ids_from_dap, **capabilities) if capabilities.__class__ != Capabilities else capabilities
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        capabilities = self.capabilities
        dct = {'capabilities': capabilities.to_dict(update_ids_to_dap=update_ids_to_dap)}
        dct.update(self.kwargs)
        return dct