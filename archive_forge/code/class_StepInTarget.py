from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class StepInTarget(BaseSchema):
    """
    A StepInTarget can be used in the 'stepIn' request and determines into which single target the
    stepIn request should step.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'id': {'type': 'integer', 'description': 'Unique identifier for a stepIn target.'}, 'label': {'type': 'string', 'description': 'The name of the stepIn target (shown in the UI).'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, id, label, update_ids_from_dap=False, **kwargs):
        """
        :param integer id: Unique identifier for a stepIn target.
        :param string label: The name of the stepIn target (shown in the UI).
        """
        self.id = id
        self.label = label
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        id = self.id
        label = self.label
        dct = {'id': id, 'label': label}
        dct.update(self.kwargs)
        return dct