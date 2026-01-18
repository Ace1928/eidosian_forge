from fontTools.misc.textTools import tostr
Test if a string is valid user input and decode it to unicode string
        using ASCII encoding if it's a bytes string.
        Reject all bytes/unicode input that contains non-XML characters.
        Reject all bytes input that contains non-ASCII characters.
        