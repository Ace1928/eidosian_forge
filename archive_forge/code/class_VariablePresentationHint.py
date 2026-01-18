from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class VariablePresentationHint(BaseSchema):
    """
    Optional properties of a variable that can be used to determine how to render the variable in the
    UI.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'kind': {'description': 'The kind of variable. Before introducing additional values, try to use the listed values.', 'type': 'string', '_enum': ['property', 'method', 'class', 'data', 'event', 'baseClass', 'innerClass', 'interface', 'mostDerivedClass', 'virtual', 'dataBreakpoint'], 'enumDescriptions': ['Indicates that the object is a property.', 'Indicates that the object is a method.', 'Indicates that the object is a class.', 'Indicates that the object is data.', 'Indicates that the object is an event.', 'Indicates that the object is a base class.', 'Indicates that the object is an inner class.', 'Indicates that the object is an interface.', 'Indicates that the object is the most derived class.', 'Indicates that the object is virtual, that means it is a synthetic object introducedby the\nadapter for rendering purposes, e.g. an index range for large arrays.', "Deprecated: Indicates that a data breakpoint is registered for the object. The 'hasDataBreakpoint' attribute should generally be used instead."]}, 'attributes': {'description': 'Set of attributes represented as an array of strings. Before introducing additional values, try to use the listed values.', 'type': 'array', 'items': {'type': 'string', '_enum': ['static', 'constant', 'readOnly', 'rawString', 'hasObjectId', 'canHaveObjectId', 'hasSideEffects', 'hasDataBreakpoint'], 'enumDescriptions': ['Indicates that the object is static.', 'Indicates that the object is a constant.', 'Indicates that the object is read only.', 'Indicates that the object is a raw string.', 'Indicates that the object can have an Object ID created for it.', 'Indicates that the object has an Object ID associated with it.', 'Indicates that the evaluation had side effects.', 'Indicates that the object has its value tracked by a data breakpoint.']}}, 'visibility': {'description': 'Visibility of variable. Before introducing additional values, try to use the listed values.', 'type': 'string', '_enum': ['public', 'private', 'protected', 'internal', 'final']}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, kind=None, attributes=None, visibility=None, update_ids_from_dap=False, **kwargs):
        """
        :param string kind: The kind of variable. Before introducing additional values, try to use the listed values.
        :param array attributes: Set of attributes represented as an array of strings. Before introducing additional values, try to use the listed values.
        :param string visibility: Visibility of variable. Before introducing additional values, try to use the listed values.
        """
        self.kind = kind
        self.attributes = attributes
        self.visibility = visibility
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        kind = self.kind
        attributes = self.attributes
        if attributes and hasattr(attributes[0], 'to_dict'):
            attributes = [x.to_dict() for x in attributes]
        visibility = self.visibility
        dct = {}
        if kind is not None:
            dct['kind'] = kind
        if attributes is not None:
            dct['attributes'] = attributes
        if visibility is not None:
            dct['visibility'] = visibility
        dct.update(self.kwargs)
        return dct