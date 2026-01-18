from pygments.style import Style
from pygments.token import Keyword, Name, Comment, String, Error, Operator

    pygments.styles.algol_nu
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Algol publication style without underlining of keywords.

    This style renders source code for publication of algorithms in
    scientific papers and academic texts, where its format is frequently used.

    It is based on the style of the revised Algol-60 language report[1].

    o  No colours, only black, white and shades of grey are used.
    o  Keywords are rendered in lowercase boldface.
    o  Builtins are rendered in lowercase boldface italic.
    o  Docstrings and pragmas are rendered in dark grey boldface.
    o  Library identifiers are rendered in dark grey boldface italic.
    o  Comments are rendered in grey italic.

    To render keywords with underlining, refer to the `Algol` style.

    For lowercase conversion of keywords and builtins in languages where
    these are not or might not be lowercase, a supporting lexer is required.
    The Algol and Modula-2 lexers automatically convert to lowercase whenever
    this style is selected.

    [1] `Revised Report on the Algorithmic Language Algol-60 <http://www.masswerk.at/algol60/report.htm>`

    :copyright: Copyright 2006-2017 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
