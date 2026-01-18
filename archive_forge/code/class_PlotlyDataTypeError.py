class PlotlyDataTypeError(PlotlyGraphObjectError):

    def __init__(self, obj, path, notes=()):
        """See PlotlyGraphObjectError.__init__ for param docs."""
        format_dict = {'index': path[-1], 'object_name': obj._name}
        message = "Invalid entry found in '{object_name}' at index, '{index}'".format(**format_dict)
        note = "It's invalid because it doesn't contain a valid 'type' value."
        notes = [note] + list(notes)
        super(PlotlyDataTypeError, self).__init__(message=message, path=path, notes=notes)