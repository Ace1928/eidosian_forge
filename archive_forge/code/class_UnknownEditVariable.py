class UnknownEditVariable(Exception):
    __slots__ = ('edit_variable',)

    def __init__(self, edit_variable):
        self.edit_variable = edit_variable