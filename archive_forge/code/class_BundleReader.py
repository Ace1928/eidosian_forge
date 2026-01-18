import bz2
import re
from io import BytesIO
import fastbencode as bencode
from .... import errors, iterablefile, lru_cache, multiparent, osutils
from .... import repository as _mod_repository
from .... import revision as _mod_revision
from .... import trace, ui
from ....i18n import ngettext
from ... import pack, serializer
from ... import versionedfile as _mod_versionedfile
from .. import bundle_data
from .. import serializer as bundle_serializer
class BundleReader:
    """Reader for bundle-format files.

    This serves roughly the same purpose as ContainerReader, but acts as a
    layer on top of it, providing metadata, a semantic name, and a record
    body
    """

    def __init__(self, fileobj, stream_input=True):
        """Constructor

        :param fileobj: a file containing a bzip-encoded container
        :param stream_input: If True, the BundleReader stream input rather than
            reading it all into memory at once.  Reading it into memory all at
            once is (currently) faster.
        """
        line = fileobj.readline()
        if line != '\n':
            fileobj.readline()
        self.patch_lines = []
        if stream_input:
            source_file = iterablefile.IterableFile(self.iter_decode(fileobj))
        else:
            source_file = BytesIO(bz2.decompress(fileobj.read()))
        self._container_file = source_file

    @staticmethod
    def iter_decode(fileobj):
        """Iterate through decoded fragments of the file"""
        decompressor = bz2.BZ2Decompressor()
        for line in fileobj:
            try:
                yield decompressor.decompress(line)
            except EOFError:
                return

    @staticmethod
    def decode_name(name):
        """Decode a name from its container form into a semantic form

        :retval: content_kind, revision_id, file_id
        """
        segments = re.split(b'(//?)', name)
        names = [b'']
        for segment in segments:
            if segment == b'//':
                names[-1] += b'/'
            elif segment == b'/':
                names.append(b'')
            else:
                names[-1] += segment
        content_kind = names[0]
        revision_id = None
        file_id = None
        if len(names) > 1:
            revision_id = names[1]
        if len(names) > 2:
            file_id = names[2]
        return (content_kind.decode('ascii'), revision_id, file_id)

    def iter_records(self):
        """Iterate through bundle records

        :return: a generator of (bytes, metadata, content_kind, revision_id,
            file_id)
        """
        iterator = pack.iter_records_from_file(self._container_file)
        for names, bytes in iterator:
            if len(names) != 1:
                raise errors.BadBundle('Record has %d names instead of 1' % len(names))
            metadata = bencode.bdecode(bytes)
            if metadata[b'storage_kind'] == b'header':
                bytes = None
            else:
                _unused, bytes = next(iterator)
            yield ((bytes, metadata) + self.decode_name(names[0][0]))