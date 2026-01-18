class ForwardAnnotationSyntaxError(Message):
    message = 'syntax error in forward annotation %r'

    def __init__(self, filename, loc, annotation):
        Message.__init__(self, filename, loc)
        self.message_args = (annotation,)