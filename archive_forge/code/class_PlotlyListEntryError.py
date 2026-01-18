class PlotlyListEntryError(PlotlyGraphObjectError):

    def __init__(self, obj, path, notes=()):
        """See PlotlyGraphObjectError.__init__ for param docs."""
        format_dict = {'index': path[-1], 'object_name': obj._name}
        message = "Invalid entry found in '{object_name}' at index, '{index}'".format(**format_dict)
        notes = [obj.help(return_help=True)] + list(notes)
        super(PlotlyListEntryError, self).__init__(message=message, path=path, notes=notes)