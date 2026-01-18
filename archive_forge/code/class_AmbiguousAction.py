class AmbiguousAction(PlexError):
    message = 'Two tokens with different actions can match the same string'

    def __init__(self):
        pass