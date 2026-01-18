from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class SourceArguments(BaseSchema):
    """
    Arguments for 'source' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'source': {'description': 'Specifies the source content to load. Either source.path or source.sourceReference must be specified.', 'type': 'Source'}, 'sourceReference': {'type': 'integer', 'description': "The reference to the source. This is the same as source.sourceReference.\nThis is provided for backward compatibility since old backends do not understand the 'source' attribute."}}
    __refs__ = set(['source'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, sourceReference, source=None, update_ids_from_dap=False, **kwargs):
        """
        :param integer sourceReference: The reference to the source. This is the same as source.sourceReference.
        This is provided for backward compatibility since old backends do not understand the 'source' attribute.
        :param Source source: Specifies the source content to load. Either source.path or source.sourceReference must be specified.
        """
        self.sourceReference = sourceReference
        if source is None:
            self.source = Source()
        else:
            self.source = Source(update_ids_from_dap=update_ids_from_dap, **source) if source.__class__ != Source else source
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        sourceReference = self.sourceReference
        source = self.source
        dct = {'sourceReference': sourceReference}
        if source is not None:
            dct['source'] = source.to_dict(update_ids_to_dap=update_ids_to_dap)
        dct.update(self.kwargs)
        return dct