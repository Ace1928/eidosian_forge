class PlotlyDictKeyError(PlotlyGraphObjectError):

    def __init__(self, obj, path, notes=()):
        """See PlotlyGraphObjectError.__init__ for param docs."""
        format_dict = {'attribute': path[-1], 'object_name': obj._name}
        message = "'{attribute}' is not allowed in '{object_name}'".format(**format_dict)
        notes = [obj.help(return_help=True)] + list(notes)
        super(PlotlyDictKeyError, self).__init__(message=message, path=path, notes=notes)