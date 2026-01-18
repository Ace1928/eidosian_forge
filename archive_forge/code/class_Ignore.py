class Ignore(Action):
    """
    IGNORE is a Plex action which causes its associated token
    to be ignored. See the docstring of Plex.Lexicon  for more
    information.
    """

    def perform(self, token_stream, text):
        return None

    def __repr__(self):
        return 'IGNORE'