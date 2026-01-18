class ImportStarNotPermitted(Message):
    message = "'from %s import *' only allowed at module level"

    def __init__(self, filename, loc, modname):
        Message.__init__(self, filename, loc)
        self.message_args = (modname,)