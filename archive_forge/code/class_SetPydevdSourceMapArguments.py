from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class SetPydevdSourceMapArguments(BaseSchema):
    """
    Arguments for 'setPydevdSourceMap' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'source': {'description': "The source location of the PydevdSourceMap; 'source.path' must be specified (e.g.: for an ipython notebook this could be something as /home/notebook/note.py).", 'type': 'Source'}, 'pydevdSourceMaps': {'type': 'array', 'items': {'$ref': '#/definitions/PydevdSourceMap'}, 'description': 'The PydevdSourceMaps to be set to the given source (provide an empty array to clear the source mappings for a given path).'}}
    __refs__ = set(['source'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, source, pydevdSourceMaps=None, update_ids_from_dap=False, **kwargs):
        """
        :param Source source: The source location of the PydevdSourceMap; 'source.path' must be specified (e.g.: for an ipython notebook this could be something as /home/notebook/note.py).
        :param array pydevdSourceMaps: The PydevdSourceMaps to be set to the given source (provide an empty array to clear the source mappings for a given path).
        """
        if source is None:
            self.source = Source()
        else:
            self.source = Source(update_ids_from_dap=update_ids_from_dap, **source) if source.__class__ != Source else source
        self.pydevdSourceMaps = pydevdSourceMaps
        if update_ids_from_dap and self.pydevdSourceMaps:
            for o in self.pydevdSourceMaps:
                PydevdSourceMap.update_dict_ids_from_dap(o)
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        source = self.source
        pydevdSourceMaps = self.pydevdSourceMaps
        if pydevdSourceMaps and hasattr(pydevdSourceMaps[0], 'to_dict'):
            pydevdSourceMaps = [x.to_dict() for x in pydevdSourceMaps]
        dct = {'source': source.to_dict(update_ids_to_dap=update_ids_to_dap)}
        if pydevdSourceMaps is not None:
            dct['pydevdSourceMaps'] = [PydevdSourceMap.update_dict_ids_to_dap(o) for o in pydevdSourceMaps] if update_ids_to_dap and pydevdSourceMaps else pydevdSourceMaps
        dct.update(self.kwargs)
        return dct