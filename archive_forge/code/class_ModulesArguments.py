from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ModulesArguments(BaseSchema):
    """
    Arguments for 'modules' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'startModule': {'type': 'integer', 'description': 'The index of the first module to return; if omitted modules start at 0.'}, 'moduleCount': {'type': 'integer', 'description': 'The number of modules to return. If moduleCount is not specified or 0, all modules are returned.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, startModule=None, moduleCount=None, update_ids_from_dap=False, **kwargs):
        """
        :param integer startModule: The index of the first module to return; if omitted modules start at 0.
        :param integer moduleCount: The number of modules to return. If moduleCount is not specified or 0, all modules are returned.
        """
        self.startModule = startModule
        self.moduleCount = moduleCount
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        startModule = self.startModule
        moduleCount = self.moduleCount
        dct = {}
        if startModule is not None:
            dct['startModule'] = startModule
        if moduleCount is not None:
            dct['moduleCount'] = moduleCount
        dct.update(self.kwargs)
        return dct