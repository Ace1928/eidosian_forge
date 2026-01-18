from collections import defaultdict
import re
from .pyutil import memoize
from .periodic import symbols
@memoize()
def _get_formula_parser():
    """Create a forward pyparsing parser for chemical formulae

    BNF for simple chemical formula (no nesting)

        integer :: '0'..'9'+
        element :: 'A'..'Z' 'a'..'z'*
        term :: element [integer]
        formula :: term+


    BNF for nested chemical formula

        integer :: '0'..'9'+
        element :: 'A'..'Z' 'a'..'z'*
        term :: (element | '(' formula ')') [integer]
        formula :: term+

    Notes
    -----
    The code in this function is from an answer on StackOverflow:
        http://stackoverflow.com/a/18555142/790973
        written by:
            Paul McGuire, http://stackoverflow.com/users/165216/paul-mcguire
        in answer to the question formulated by:
            Thales MG, http://stackoverflow.com/users/2708711/thales-mg
        the code is licensed under 'CC-WIKI'.
        (see: http://blog.stackoverflow.com/2009/06/attribution-required/)

    Documentation for the desired product.  Original documentation
    above.

    Create a chemical formula parser.

    Parse a chemical formula, including elements, nested ions,
    complexes, charges (ions), hydrates, and state symbols.

    BNF for nested chemical formula with complexes

        count :: ( '1'..'9'? | '1'..'9'' '0'..'9'+ )
        element :: 'A'..'Z' 'a'..'z'*
        charge :: ( '-' | '+' ) ( '1'..'9'? | '1'..'9'' '0'..'9'+ )
        prime :: ( "*" | "'" )*
        term :: (element
                 | '(' formula ')'
                 | '{' formula '}'
                 | '[' formula ']' ) count prime charge?
        formula :: term+
        hydrate :: '.' count? formula
        state :: '(' ( 's' | 'l' | 'g' | 'aq' | 'cr' ) ')'
        compound :: count formula hydrate? state?

    Parse a chemical formula, including elements, non-integer
    subscripts, nested ions, complexes, charges (ions), hydrates, and
    state symbols.

    BNF for nested chemical formula with complexes

        count :: ( '1'..'9'? | '1'..'9'' '0'..'9'+ | '0'..'9'+ '.' '0'..'9'+ )
        element :: 'A'..'Z' 'a'..'z'*
        charge :: ( '-' | '+' ) ( '1'..'9'? | '1'..'9'' '0'..'9'+ )
        prime :: ( "*" | "'" )*
        term :: (element
                 | '(' formula ')'
                 | '{' formula '}'
                 | '[' formula ']' ) count prime charge?
        formula :: term+
        hydrate :: '..' count? formula
        state :: '(' ( 's' | 'l' | 'g' | 'aq' | 'cr' ) ')'
        compound :: count formula hydrate? state?
    """
    _p = __import__(parsing_library)
    Forward, Group, OneOrMore = (_p.Forward, _p.Group, _p.OneOrMore)
    Optional, ParseResults, Regex = (_p.Optional, _p.ParseResults, _p.Regex)
    Suppress = _p.Suppress
    LCB = Suppress(Regex('\\{'))
    RCB = Suppress(Regex('\\}'))
    LSB = Suppress(Regex('\\['))
    RSB = Suppress(Regex('\\]'))
    LP = Suppress(Regex('\\('))
    RP = Suppress(Regex('\\)'))
    caged = Suppress(Regex('\\@'))
    primes = Suppress(Regex("[*']+"))
    count = Regex('(\\d+\\.\\d+|\\d*)')
    count.setParseAction(lambda t: 1 if t[0] == '' else float(t[0]))
    state = Suppress(Regex('\\((s|l|g|aq|cr)\\)'))
    element = Regex('A[cglmrstu]|B[aehikr]?|C[adeflmnorsu]?|D[bsy]|E[rsu]|F[elmr]?|G[ade]|H[efgos]?|I[nr]?|Kr?|L[airuv]|M[cdgnot]|N[abdehiop]?|O[gs]?|P[abdmortu]?|R[abefghnu]|S[bcegimnr]?|T[abcehilms]|U|V|W|Xe|Yb?|Z[nr]').setResultsName('element', listAllMatches=True)
    formula = Forward()
    term = Group((element | Group(LP + formula + RP)('subgroup') | Group(LSB + formula + RSB)('subgroup') | Group(LCB + formula + RCB)('subgroup') | Group(caged + formula)('subgroup')) + Optional(count, default=1)('mult') + Optional(state)('state') + Optional(primes)('primes'))

    def multiplyContents(tokens):
        t = tokens[0]
        if t.subgroup:
            mult = t.mult
            for term in t.subgroup:
                term[1] *= mult
            return t.subgroup
    term.setParseAction(multiplyContents)

    def sumByElement(tokens):
        elementsList = [t[0] for t in tokens]
        duplicates = len(elementsList) > len(set(elementsList))
        if duplicates:
            ctr = defaultdict(int)
            for t in tokens:
                ctr[t[0]] += t[1]
            return ParseResults([ParseResults([k, v]) for k, v in ctr.items()])
    formula << OneOrMore(term)
    formula.setParseAction(sumByElement)
    return formula