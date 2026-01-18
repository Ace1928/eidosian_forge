class UndefinedLocal(Message):
    message = 'local variable %r {0} referenced before assignment'
    default = 'defined in enclosing scope on line %r'
    builtin = 'defined as a builtin'

    def __init__(self, filename, loc, name, orig_loc):
        Message.__init__(self, filename, loc)
        if orig_loc is None:
            self.message = self.message.format(self.builtin)
            self.message_args = name
        else:
            self.message = self.message.format(self.default)
            self.message_args = (name, orig_loc.lineno)