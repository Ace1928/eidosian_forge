import operator
from debian._deb822_repro._util import BufferingIterator
from debian._deb822_repro.tokens import Deb822Token
The actual context of the token

        This field *must not* be used to determine the type of token.  The
        formatter cannot reliably tell whether "#..." is a comment or a value
        (it can be both).  Use is_value and is_comment instead for discriminating
        token types.

        For value tokens, this the concrete value to be omitted.

        For comment token, this is the full comment text.

        This is the same as str(token).
        