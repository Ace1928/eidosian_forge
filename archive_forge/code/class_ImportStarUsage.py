class ImportStarUsage(Message):
    message = '%r may be undefined, or defined from star imports: %s'

    def __init__(self, filename, loc, name, from_list):
        Message.__init__(self, filename, loc)
        self.message_args = (name, from_list)