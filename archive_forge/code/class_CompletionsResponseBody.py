from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class CompletionsResponseBody(BaseSchema):
    """
    "body" of CompletionsResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'targets': {'type': 'array', 'items': {'$ref': '#/definitions/CompletionItem'}, 'description': 'The possible completions for .'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, targets, update_ids_from_dap=False, **kwargs):
        """
        :param array targets: The possible completions for .
        """
        self.targets = targets
        if update_ids_from_dap and self.targets:
            for o in self.targets:
                CompletionItem.update_dict_ids_from_dap(o)
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        targets = self.targets
        if targets and hasattr(targets[0], 'to_dict'):
            targets = [x.to_dict() for x in targets]
        dct = {'targets': [CompletionItem.update_dict_ids_to_dap(o) for o in targets] if update_ids_to_dap and targets else targets}
        dct.update(self.kwargs)
        return dct