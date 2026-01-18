from struct import unpack
A union is encoded by first writing a long value indicating the
        zero-based position within the union of the schema of its value.

        The value is then encoded per the indicated schema within the union.
        