from nltk.parse import MaltParser
from nltk.sem.drt import DrsDrawer, DrtVariableExpression
from nltk.sem.glue import DrtGlue
from nltk.sem.logic import Variable
from nltk.tag import RegexpTagger
from nltk.util import in_idle
def _exampleList_store_selection(self, index):
    self._curExample = index
    example = self._examples[index]
    self._exampleList.selection_clear(0, 'end')
    if example:
        cache = self._readingCache[index]
        if cache:
            if isinstance(cache, list):
                self._readings = cache
                self._error = None
            else:
                self._readings = []
                self._error = cache
        else:
            try:
                self._readings = self._glue.parse_to_meaning(example)
                self._error = None
                self._readingCache[index] = self._readings
            except Exception as e:
                self._readings = []
                self._error = DrtVariableExpression(Variable('Error: ' + str(e)))
                self._readingCache[index] = self._error
                self._exampleList.delete(index)
                self._exampleList.insert(index, '  %s *' % example)
                self._exampleList.config(height=min(len(self._examples), 25), width=40)
        self._populate_readingListbox()
        self._exampleList.selection_set(index)
        self._drs = None
        self._redraw()