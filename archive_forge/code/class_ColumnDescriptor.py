from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ColumnDescriptor(BaseSchema):
    """
    A ColumnDescriptor specifies what module attribute to show in a column of the ModulesView, how to
    format it,
    
    and what the column's label should be.
    
    It is only used if the underlying UI actually supports this level of customization.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'attributeName': {'type': 'string', 'description': 'Name of the attribute rendered in this column.'}, 'label': {'type': 'string', 'description': 'Header UI label of column.'}, 'format': {'type': 'string', 'description': 'Format to use for the rendered values in this column. TBD how the format strings looks like.'}, 'type': {'type': 'string', 'enum': ['string', 'number', 'boolean', 'unixTimestampUTC'], 'description': "Datatype of values in this column.  Defaults to 'string' if not specified."}, 'width': {'type': 'integer', 'description': 'Width of this column in characters (hint only).'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, attributeName, label, format=None, type=None, width=None, update_ids_from_dap=False, **kwargs):
        """
        :param string attributeName: Name of the attribute rendered in this column.
        :param string label: Header UI label of column.
        :param string format: Format to use for the rendered values in this column. TBD how the format strings looks like.
        :param string type: Datatype of values in this column.  Defaults to 'string' if not specified.
        :param integer width: Width of this column in characters (hint only).
        """
        self.attributeName = attributeName
        self.label = label
        self.format = format
        self.type = type
        self.width = width
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        attributeName = self.attributeName
        label = self.label
        format = self.format
        type = self.type
        width = self.width
        dct = {'attributeName': attributeName, 'label': label}
        if format is not None:
            dct['format'] = format
        if type is not None:
            dct['type'] = type
        if width is not None:
            dct['width'] = width
        dct.update(self.kwargs)
        return dct