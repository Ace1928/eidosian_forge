import sys
from enchant.checker import SpellChecker
Run spellchecking on the named file.
        This method can be used to run the spellchecker over the named file.
        If <outfile> is not given, the corrected contents replace the contents
        of <infile>.  If <outfile> is given, the corrected contents will be
        written to that file.  Use "-" to have the contents written to stdout.
        If <enc> is given, it specifies the encoding used to read the
        file's contents into a unicode string.  The output will be written
        in the same encoding.
        