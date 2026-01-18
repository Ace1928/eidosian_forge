from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ExceptionBreakpointsFilter(BaseSchema):
    """
    An ExceptionBreakpointsFilter is shown in the UI as an filter option for configuring how exceptions
    are dealt with.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'filter': {'type': 'string', 'description': "The internal ID of the filter option. This value is passed to the 'setExceptionBreakpoints' request."}, 'label': {'type': 'string', 'description': 'The name of the filter option. This will be shown in the UI.'}, 'description': {'type': 'string', 'description': 'An optional help text providing additional information about the exception filter. This string is typically shown as a hover and must be translated.'}, 'default': {'type': 'boolean', 'description': "Initial value of the filter option. If not specified a value 'false' is assumed."}, 'supportsCondition': {'type': 'boolean', 'description': 'Controls whether a condition can be specified for this filter option. If false or missing, a condition can not be set.'}, 'conditionDescription': {'type': 'string', 'description': 'An optional help text providing information about the condition. This string is shown as the placeholder text for a text box and must be translated.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, filter, label, description=None, default=None, supportsCondition=None, conditionDescription=None, update_ids_from_dap=False, **kwargs):
        """
        :param string filter: The internal ID of the filter option. This value is passed to the 'setExceptionBreakpoints' request.
        :param string label: The name of the filter option. This will be shown in the UI.
        :param string description: An optional help text providing additional information about the exception filter. This string is typically shown as a hover and must be translated.
        :param boolean default: Initial value of the filter option. If not specified a value 'false' is assumed.
        :param boolean supportsCondition: Controls whether a condition can be specified for this filter option. If false or missing, a condition can not be set.
        :param string conditionDescription: An optional help text providing information about the condition. This string is shown as the placeholder text for a text box and must be translated.
        """
        self.filter = filter
        self.label = label
        self.description = description
        self.default = default
        self.supportsCondition = supportsCondition
        self.conditionDescription = conditionDescription
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        filter = self.filter
        label = self.label
        description = self.description
        default = self.default
        supportsCondition = self.supportsCondition
        conditionDescription = self.conditionDescription
        dct = {'filter': filter, 'label': label}
        if description is not None:
            dct['description'] = description
        if default is not None:
            dct['default'] = default
        if supportsCondition is not None:
            dct['supportsCondition'] = supportsCondition
        if conditionDescription is not None:
            dct['conditionDescription'] = conditionDescription
        dct.update(self.kwargs)
        return dct