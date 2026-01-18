import re
import html
Converts a string in CIF text-format to unicode.  Any HTML tags
    contained in the string are removed.  HTML numeric character references
    are unescaped (i.e. converted to unicode).

    Parameters:

    s: string
        The CIF text string to convert

    Returns:

    u: string
        A unicode formatted string.
    