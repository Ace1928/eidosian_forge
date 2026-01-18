import re,sys
from pyparsing import Word, alphas, ParseException, Literal, CaselessLiteral \

   Tests the parsing of various supported expressions. Raises
   an AssertError if the output is not what is expected. Prints the
   input, expected output, and actual output for all tests.
   