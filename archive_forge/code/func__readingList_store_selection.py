from nltk.parse import MaltParser
from nltk.sem.drt import DrsDrawer, DrtVariableExpression
from nltk.sem.glue import DrtGlue
from nltk.sem.logic import Variable
from nltk.tag import RegexpTagger
from nltk.util import in_idle
def _readingList_store_selection(self, index):
    reading = self._readings[index]
    self._readingList.selection_clear(0, 'end')
    if reading:
        self._readingList.selection_set(index)
        self._drs = reading.simplify().normalize().resolve_anaphora()
        self._redraw()