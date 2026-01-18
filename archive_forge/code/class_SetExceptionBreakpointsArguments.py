from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class SetExceptionBreakpointsArguments(BaseSchema):
    """
    Arguments for 'setExceptionBreakpoints' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'filters': {'type': 'array', 'items': {'type': 'string'}, 'description': "Set of exception filters specified by their ID. The set of all possible exception filters is defined by the 'exceptionBreakpointFilters' capability. The 'filter' and 'filterOptions' sets are additive."}, 'filterOptions': {'type': 'array', 'items': {'$ref': '#/definitions/ExceptionFilterOptions'}, 'description': "Set of exception filters and their options. The set of all possible exception filters is defined by the 'exceptionBreakpointFilters' capability. This attribute is only honored by a debug adapter if the capability 'supportsExceptionFilterOptions' is true. The 'filter' and 'filterOptions' sets are additive."}, 'exceptionOptions': {'type': 'array', 'items': {'$ref': '#/definitions/ExceptionOptions'}, 'description': "Configuration options for selected exceptions.\nThe attribute is only honored by a debug adapter if the capability 'supportsExceptionOptions' is true."}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, filters, filterOptions=None, exceptionOptions=None, update_ids_from_dap=False, **kwargs):
        """
        :param array filters: Set of exception filters specified by their ID. The set of all possible exception filters is defined by the 'exceptionBreakpointFilters' capability. The 'filter' and 'filterOptions' sets are additive.
        :param array filterOptions: Set of exception filters and their options. The set of all possible exception filters is defined by the 'exceptionBreakpointFilters' capability. This attribute is only honored by a debug adapter if the capability 'supportsExceptionFilterOptions' is true. The 'filter' and 'filterOptions' sets are additive.
        :param array exceptionOptions: Configuration options for selected exceptions.
        The attribute is only honored by a debug adapter if the capability 'supportsExceptionOptions' is true.
        """
        self.filters = filters
        self.filterOptions = filterOptions
        if update_ids_from_dap and self.filterOptions:
            for o in self.filterOptions:
                ExceptionFilterOptions.update_dict_ids_from_dap(o)
        self.exceptionOptions = exceptionOptions
        if update_ids_from_dap and self.exceptionOptions:
            for o in self.exceptionOptions:
                ExceptionOptions.update_dict_ids_from_dap(o)
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        filters = self.filters
        if filters and hasattr(filters[0], 'to_dict'):
            filters = [x.to_dict() for x in filters]
        filterOptions = self.filterOptions
        if filterOptions and hasattr(filterOptions[0], 'to_dict'):
            filterOptions = [x.to_dict() for x in filterOptions]
        exceptionOptions = self.exceptionOptions
        if exceptionOptions and hasattr(exceptionOptions[0], 'to_dict'):
            exceptionOptions = [x.to_dict() for x in exceptionOptions]
        dct = {'filters': filters}
        if filterOptions is not None:
            dct['filterOptions'] = [ExceptionFilterOptions.update_dict_ids_to_dap(o) for o in filterOptions] if update_ids_to_dap and filterOptions else filterOptions
        if exceptionOptions is not None:
            dct['exceptionOptions'] = [ExceptionOptions.update_dict_ids_to_dap(o) for o in exceptionOptions] if update_ids_to_dap and exceptionOptions else exceptionOptions
        dct.update(self.kwargs)
        return dct