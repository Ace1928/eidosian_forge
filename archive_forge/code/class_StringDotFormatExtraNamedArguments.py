class StringDotFormatExtraNamedArguments(Message):
    message = "'...'.format(...) has unused named argument(s): %s"

    def __init__(self, filename, loc, extra_keywords):
        Message.__init__(self, filename, loc)
        self.message_args = (extra_keywords,)