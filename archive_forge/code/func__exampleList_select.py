from nltk.parse import MaltParser
from nltk.sem.drt import DrsDrawer, DrtVariableExpression
from nltk.sem.glue import DrtGlue
from nltk.sem.logic import Variable
from nltk.tag import RegexpTagger
from nltk.util import in_idle
def _exampleList_select(self, event):
    selection = self._exampleList.curselection()
    if len(selection) != 1:
        return
    self._exampleList_store_selection(int(selection[0]))