import re
from io import BytesIO
from .. import errors
class ContainerReader(BaseReader):
    """A class for reading Bazaar's container format."""

    def iter_records(self):
        """Iterate over the container, yielding each record as it is read.

        Each yielded record will be a 2-tuple of (names, callable), where names
        is a ``list`` and bytes is a function that takes one argument,
        ``max_length``.

        You **must not** call the callable after advancing the iterator to the
        next record.  That is, this code is invalid::

            record_iter = container.iter_records()
            names1, callable1 = record_iter.next()
            names2, callable2 = record_iter.next()
            bytes1 = callable1(None)

        As it will give incorrect results and invalidate the state of the
        ContainerReader.

        :raises ContainerError: if any sort of container corruption is
            detected, e.g. UnknownContainerFormatError is the format of the
            container is unrecognised.
        :seealso: ContainerReader.read
        """
        self._read_format()
        return self._iter_records()

    def iter_record_objects(self):
        """Iterate over the container, yielding each record as it is read.

        Each yielded record will be an object with ``read`` and ``validate``
        methods.  Like with iter_records, it is not safe to use a record object
        after advancing the iterator to yield next record.

        :raises ContainerError: if any sort of container corruption is
            detected, e.g. UnknownContainerFormatError is the format of the
            container is unrecognised.
        :seealso: iter_records
        """
        self._read_format()
        return self._iter_record_objects()

    def _iter_records(self):
        for record in self._iter_record_objects():
            yield record.read()

    def _iter_record_objects(self):
        while True:
            try:
                record_kind = self.reader_func(1)
            except StopIteration:
                return
            if record_kind == b'B':
                reader = BytesRecordReader(self._source)
                yield reader
            elif record_kind == b'E':
                return
            elif record_kind == b'':
                raise UnexpectedEndOfContainerError()
            else:
                raise UnknownRecordTypeError(record_kind)

    def _read_format(self):
        format = self._read_line()
        if format != FORMAT_ONE:
            raise UnknownContainerFormatError(format)

    def validate(self):
        """Validate this container and its records.

        Validating consumes the data stream just like iter_records and
        iter_record_objects, so you cannot call it after
        iter_records/iter_record_objects.

        :raises ContainerError: if something is invalid.
        """
        all_names = set()
        for record_names, read_bytes in self.iter_records():
            read_bytes(None)
            for name_tuple in record_names:
                for name in name_tuple:
                    _check_name_encoding(name)
                if name_tuple in all_names:
                    raise DuplicateRecordNameError(name_tuple[0])
                all_names.add(name_tuple)
        excess_bytes = self.reader_func(1)
        if excess_bytes != b'':
            raise ContainerHasExcessDataError(excess_bytes)