from typing import List
from redis import DataError
class TagField(Field):
    """
    TagField is a tag-indexing field with simpler compression and tokenization.
    See http://redisearch.io/Tags/
    """
    SEPARATOR = 'SEPARATOR'
    CASESENSITIVE = 'CASESENSITIVE'

    def __init__(self, name: str, separator: str=',', case_sensitive: bool=False, withsuffixtrie: bool=False, **kwargs):
        args = [Field.TAG, self.SEPARATOR, separator]
        if case_sensitive:
            args.append(self.CASESENSITIVE)
        if withsuffixtrie:
            args.append('WITHSUFFIXTRIE')
        Field.__init__(self, name, args=args, **kwargs)