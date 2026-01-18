class PercentFormatMissingArgument(Message):
    message = "'...' %% ... is missing argument(s) for placeholder(s): %s"

    def __init__(self, filename, loc, missing_arguments):
        Message.__init__(self, filename, loc)
        self.message_args = (missing_arguments,)