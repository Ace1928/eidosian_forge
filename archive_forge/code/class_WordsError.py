class WordsError(Exception):

    def __str__(self) -> str:
        return self.__class__.__name__ + ': ' + Exception.__str__(self)