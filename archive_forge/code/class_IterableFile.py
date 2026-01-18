from typing import Callable, Iterable, Iterator, List, Optional
class IterableFile:
    """This class supplies all File methods that can be implemented cheaply."""

    def __init__(self, iterable: Iterable[bytes]) -> None:
        object.__init__(self)
        self._file_base = IterableFileBase(iterable)
        self._iter = self._make_iterator()
        self._closed = False
        self.softspace = 0

    def _make_iterator(self) -> Iterator[bytes]:
        while not self._file_base.done:
            self._check_closed()
            result = self._file_base.read_to(b'\n')
            if result != b'':
                yield result

    def _check_closed(self):
        if self.closed:
            raise ValueError('File is closed.')

    def close(self) -> None:
        """
        >>> f = IterableFile([b'This ', b'is ', b'a ', b'test.'])
        >>> f.closed
        False
        >>> f.close()
        >>> f.closed
        True
        """
        self._file_base.done = True
        self._closed = True
    closed = property(lambda x: x._closed)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.close()
        except BaseException:
            if exc_type is None:
                raise
        return False

    def flush(self) -> None:
        """No-op for standard compliance.
        >>> f = IterableFile([])
        >>> f.close()
        >>> f.flush()
        Traceback (most recent call last):
        ValueError: File is closed.
        """
        self._check_closed()

    def __next__(self) -> bytes:
        """Implementation of the iterator protocol's next()

        >>> f = IterableFile([b'This \\n', b'is ', b'a ', b'test.'])
        >>> next(f)
        b'This \\n'
        >>> f.close()
        >>> next(f)
        Traceback (most recent call last):
        ValueError: File is closed.
        >>> f = IterableFile([b'This \\n', b'is ', b'a ', b'test.\\n'])
        >>> next(f)
        b'This \\n'
        >>> next(f)
        b'is a test.\\n'
        >>> next(f)
        Traceback (most recent call last):
        StopIteration
        """
        self._check_closed()
        return next(self._iter)

    def __iter__(self) -> Iterator[bytes]:
        """
        >>> list(IterableFile([b'Th\\nis ', b'is \\n', b'a ', b'te\\nst.']))
        [b'Th\\n', b'is is \\n', b'a te\\n', b'st.']
        >>> f = IterableFile([b'Th\\nis ', b'is \\n', b'a ', b'te\\nst.'])
        >>> f.close()
        >>> list(f)
        Traceback (most recent call last):
        ValueError: File is closed.
        """
        return self

    def read(self, length: Optional[int]=None) -> bytes:
        """
        >>> IterableFile([b'This ', b'is ', b'a ', b'test.']).read()
        b'This is a test.'
        >>> f = IterableFile([b'This ', b'is ', b'a ', b'test.'])
        >>> f.read(10)
        b'This is a '
        >>> f = IterableFile([b'This ', b'is ', b'a ', b'test.'])
        >>> f.close()
        >>> f.read(10)
        Traceback (most recent call last):
        ValueError: File is closed.
        """
        self._check_closed()
        if length is None:
            return self._file_base.read_all()
        else:
            return self._file_base.read_n(length)

    def read_to(self, sequence: bytes, size: Optional[int]=None) -> bytes:
        """
        Read characters until a sequence is found, with optional max size.
        The specified sequence, if found, will be included in the result

        >>> f = IterableFile([b'Th\\nis ', b'is \\n', b'a ', b'te\\nst.'])
        >>> f.read_to(b'i')
        b'Th\\ni'
        >>> f.read_to(b'i')
        b's i'
        >>> f.close()
        >>> f.read_to(b'i')
        Traceback (most recent call last):
        ValueError: File is closed.
        """
        self._check_closed()
        return self._file_base.read_to(sequence, size)

    def readline(self, size: Optional[int]=None) -> bytes:
        """
        >>> f = IterableFile([b'Th\\nis ', b'is \\n', b'a ', b'te\\nst.'])
        >>> f.readline()
        b'Th\\n'
        >>> f.readline(4)
        b'is i'
        >>> f.close()
        >>> f.readline()
        Traceback (most recent call last):
        ValueError: File is closed.
        """
        return self.read_to(b'\n', size)

    def readlines(self, sizehint=None) -> List[bytes]:
        """
        >>> f = IterableFile([b'Th\\nis ', b'is \\n', b'a ', b'te\\nst.'])
        >>> f.readlines()
        [b'Th\\n', b'is is \\n', b'a te\\n', b'st.']
        >>> f = IterableFile([b'Th\\nis ', b'is \\n', b'a ', b'te\\nst.'])
        >>> f.close()
        >>> f.readlines()
        Traceback (most recent call last):
        ValueError: File is closed.
        """
        lines: List[bytes] = []
        while True:
            line = self.readline()
            if line == b'':
                return lines
            if sizehint is None:
                lines.append(line)
            elif len(line) < sizehint:
                lines.append(line)
                sizehint -= len(line)
            else:
                self._file_base.push_back(line)
                return lines