from typing import Callable, Iterable, Iterator, List, Optional
class IterableFileBase:
    """Create a file-like object from any iterable"""

    def __init__(self, iterable: Iterable[bytes]) -> None:
        object.__init__(self)
        self._iter = iter(iterable)
        self._buffer = b''
        self.done = False

    def read_n(self, length: int) -> bytes:
        """
        >>> IterableFileBase([b'This ', b'is ', b'a ', b'test.']).read_n(8)
        b'This is '
        """

        def test_length(result):
            if len(result) >= length:
                return length
            else:
                return None
        return self._read(test_length)

    def read_to(self, sequence: bytes, length: Optional[int]=None) -> bytes:
        """
        >>> f = IterableFileBase([b'Th\\nis ', b'is \\n', b'a ', b'te\\nst.'])
        >>> f.read_to(b'\\n')
        b'Th\\n'
        >>> f.read_to(b'\\n')
        b'is is \\n'
        """

        def test_contents(result):
            if length is not None:
                if len(result) >= length:
                    return length
            try:
                return result.index(sequence) + len(sequence)
            except ValueError:
                return None
        return self._read(test_contents)

    def _read(self, result_length: Callable[[bytes], Optional[int]]) -> bytes:
        """
        Read data until result satisfies the condition result_length.
        result_length is a callable that returns None until the condition
        is satisfied, and returns the length of the result to use when
        the condition is satisfied.  (i.e. it returns the length of the
        subset of the first condition match.)
        """
        result = self._buffer
        while result_length(result) is None:
            try:
                result += next(self._iter)
            except StopIteration:
                self.done = True
                self._buffer = b''
                return result
        output_length = result_length(result)
        self._buffer = result[output_length:]
        return result[:output_length]

    def read_all(self) -> bytes:
        """
        >>> IterableFileBase([b'This ', b'is ', b'a ', b'test.']).read_all()
        b'This is a test.'
        """

        def no_stop(result):
            return None
        return self._read(no_stop)

    def push_back(self, contents: bytes) -> None:
        """
        >>> f = IterableFileBase([b'Th\\nis ', b'is \\n', b'a ', b'te\\nst.'])
        >>> f.read_to(b'\\n')
        b'Th\\n'
        >>> f.push_back(b"Sh")
        >>> f.read_all()
        b'Shis is \\na te\\nst.'
        """
        self._buffer = contents + self._buffer