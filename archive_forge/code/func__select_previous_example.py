from nltk.parse import MaltParser
from nltk.sem.drt import DrsDrawer, DrtVariableExpression
from nltk.sem.glue import DrtGlue
from nltk.sem.logic import Variable
from nltk.tag import RegexpTagger
from nltk.util import in_idle
def _select_previous_example(self):
    if self._curExample > 0:
        self._exampleList_store_selection(self._curExample - 1)
    else:
        self._exampleList_store_selection(len(self._examples) - 1)