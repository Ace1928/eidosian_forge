import io
Read a line from this buffer efficiently.

        A line is a contiguous sequence of bytes that ends with either:

        1. The ``terminator`` character
        2. The end of the buffer itself

        :param byte terminator: The line terminator character.
        :rtype: bytes

        