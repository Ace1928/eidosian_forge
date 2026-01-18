from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ModuleEventBody(BaseSchema):
    """
    "body" of ModuleEvent

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'reason': {'type': 'string', 'description': 'The reason for the event.', 'enum': ['new', 'changed', 'removed']}, 'module': {'description': "The new, changed, or removed module. In case of 'removed' only the module id is used.", 'type': 'Module'}}
    __refs__ = set(['module'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, reason, module, update_ids_from_dap=False, **kwargs):
        """
        :param string reason: The reason for the event.
        :param Module module: The new, changed, or removed module. In case of 'removed' only the module id is used.
        """
        self.reason = reason
        if module is None:
            self.module = Module()
        else:
            self.module = Module(update_ids_from_dap=update_ids_from_dap, **module) if module.__class__ != Module else module
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        reason = self.reason
        module = self.module
        dct = {'reason': reason, 'module': module.to_dict(update_ids_to_dap=update_ids_to_dap)}
        dct.update(self.kwargs)
        return dct