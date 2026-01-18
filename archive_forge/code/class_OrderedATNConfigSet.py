from io import StringIO
from functools import reduce
from antlr4.PredictionContext import PredictionContext, merge
from antlr4.Utils import str_list
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.SemanticContext import SemanticContext
from antlr4.error.Errors import UnsupportedOperationException, IllegalStateException
class OrderedATNConfigSet(ATNConfigSet):

    def __init__(self):
        super().__init__()