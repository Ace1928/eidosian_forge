from typing import List
from redis import DataError
class TextField(Field):
    """
    TextField is used to define a text field in a schema definition
    """
    NOSTEM = 'NOSTEM'
    PHONETIC = 'PHONETIC'

    def __init__(self, name: str, weight: float=1.0, no_stem: bool=False, phonetic_matcher: str=None, withsuffixtrie: bool=False, **kwargs):
        Field.__init__(self, name, args=[Field.TEXT, Field.WEIGHT, weight], **kwargs)
        if no_stem:
            Field.append_arg(self, self.NOSTEM)
        if phonetic_matcher and phonetic_matcher in ['dm:en', 'dm:fr', 'dm:pt', 'dm:es']:
            Field.append_arg(self, self.PHONETIC)
            Field.append_arg(self, phonetic_matcher)
        if withsuffixtrie:
            Field.append_arg(self, 'WITHSUFFIXTRIE')