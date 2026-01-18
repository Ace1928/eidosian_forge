class FutureFeatureNotDefined(Message):
    """An undefined __future__ feature name was imported."""
    message = 'future feature %s is not defined'

    def __init__(self, filename, loc, name):
        Message.__init__(self, filename, loc)
        self.message_args = (name,)