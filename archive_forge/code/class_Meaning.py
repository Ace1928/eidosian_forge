import os
import sqlite3
from nltk.corpus.reader.api import CorpusReader
class Meaning(dict):
    """
    Represents a single PanLex meaning. A meaning is a translation set derived
    from a single source.
    """

    def __init__(self, mn, attr):
        super().__init__(**attr)
        self['mn'] = mn

    def id(self):
        """
        :return: the meaning's id.
        :rtype: int
        """
        return self['mn']

    def quality(self):
        """
        :return: the meaning's source's quality (0=worst, 9=best).
        :rtype: int
        """
        return self['uq']

    def source(self):
        """
        :return: the meaning's source id.
        :rtype: int
        """
        return self['ap']

    def source_group(self):
        """
        :return: the meaning's source group id.
        :rtype: int
        """
        return self['ui']

    def expressions(self):
        """
        :return: the meaning's expressions as a dictionary whose keys are language
            variety uniform identifiers and whose values are lists of expression
            texts.
        :rtype: dict
        """
        return self['ex']