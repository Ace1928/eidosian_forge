class UndefinedName(Message):
    message = 'undefined name %r'

    def __init__(self, filename, loc, name):
        Message.__init__(self, filename, loc)
        self.message_args = (name,)