import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
class MODE:
    BlockStatement, Statement, ObjectLiteral, ArrayLiteral, ForInitializer, Conditional, Expression = range(7)