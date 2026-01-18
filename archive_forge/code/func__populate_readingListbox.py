from nltk.parse import MaltParser
from nltk.sem.drt import DrsDrawer, DrtVariableExpression
from nltk.sem.glue import DrtGlue
from nltk.sem.logic import Variable
from nltk.tag import RegexpTagger
from nltk.util import in_idle
def _populate_readingListbox(self):
    self._readingList.delete(0, 'end')
    for i in range(len(self._readings)):
        self._readingList.insert('end', '  %s' % (i + 1))
    self._readingList.config(height=min(len(self._readings), 25), width=5)
    self._readingList.bind('<<ListboxSelect>>', self._readingList_select)