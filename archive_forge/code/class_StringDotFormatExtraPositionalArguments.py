class StringDotFormatExtraPositionalArguments(Message):
    message = "'...'.format(...) has unused arguments at position(s): %s"

    def __init__(self, filename, loc, extra_positions):
        Message.__init__(self, filename, loc)
        self.message_args = (extra_positions,)