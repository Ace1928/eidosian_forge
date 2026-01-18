from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ExceptionPathSegment(BaseSchema):
    """
    An ExceptionPathSegment represents a segment in a path that is used to match leafs or nodes in a
    tree of exceptions.
    
    If a segment consists of more than one name, it matches the names provided if 'negate' is false or
    missing or
    
    it matches anything except the names provided if 'negate' is true.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'negate': {'type': 'boolean', 'description': 'If false or missing this segment matches the names provided, otherwise it matches anything except the names provided.'}, 'names': {'type': 'array', 'items': {'type': 'string'}, 'description': "Depending on the value of 'negate' the names that should match or not match."}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, names, negate=None, update_ids_from_dap=False, **kwargs):
        """
        :param array names: Depending on the value of 'negate' the names that should match or not match.
        :param boolean negate: If false or missing this segment matches the names provided, otherwise it matches anything except the names provided.
        """
        self.names = names
        self.negate = negate
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        names = self.names
        if names and hasattr(names[0], 'to_dict'):
            names = [x.to_dict() for x in names]
        negate = self.negate
        dct = {'names': names}
        if negate is not None:
            dct['negate'] = negate
        dct.update(self.kwargs)
        return dct