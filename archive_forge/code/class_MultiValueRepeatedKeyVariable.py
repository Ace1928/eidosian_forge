class MultiValueRepeatedKeyVariable(Message):
    message = 'dictionary key variable %s repeated with different values'

    def __init__(self, filename, loc, key):
        Message.__init__(self, filename, loc)
        self.message_args = (key,)