from ... import types as sqltypes
class JSONPathType(_FormatTypeMixin, sqltypes.JSON.JSONPathType):

    def _format_value(self, value):
        return '$%s' % ''.join(['[%s]' % elem if isinstance(elem, int) else '."%s"' % elem for elem in value])