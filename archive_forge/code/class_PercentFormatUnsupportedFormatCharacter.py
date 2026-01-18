class PercentFormatUnsupportedFormatCharacter(Message):
    message = "'...' %% ... has unsupported format character %r"

    def __init__(self, filename, loc, c):
        Message.__init__(self, filename, loc)
        self.message_args = (c,)