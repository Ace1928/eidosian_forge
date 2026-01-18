from nltk.parse import MaltParser
from nltk.sem.drt import DrsDrawer, DrtVariableExpression
from nltk.sem.glue import DrtGlue
from nltk.sem.logic import Variable
from nltk.tag import RegexpTagger
from nltk.util import in_idle
def _init_glue(self):
    tagger = RegexpTagger([('^(David|Mary|John)$', 'NNP'), ('^(walks|sees|eats|chases|believes|gives|sleeps|chases|persuades|tries|seems|leaves)$', 'VB'), ('^(go|order|vanish|find|approach)$', 'VB'), ('^(a)$', 'ex_quant'), ('^(every)$', 'univ_quant'), ('^(sandwich|man|dog|pizza|unicorn|cat|senator)$', 'NN'), ('^(big|gray|former)$', 'JJ'), ('^(him|himself)$', 'PRP')])
    depparser = MaltParser(tagger=tagger)
    self._glue = DrtGlue(depparser=depparser, remove_duplicates=False)