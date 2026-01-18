import re
import warnings
from itertools import chain
from xml.etree import ElementTree
from xml.sax.saxutils import XMLGenerator, escape
from Bio import BiopythonParserWarning
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
class BlastXmlIndexer(SearchIndexer):
    """Indexer class for BLAST XML output."""
    _parser = BlastXmlParser
    qstart_mark = b'<Iteration>'
    qend_mark = b'</Iteration>'
    block_size = 16384

    def __init__(self, filename, **kwargs):
        """Initialize the class."""
        SearchIndexer.__init__(self, filename)
        iter_obj = self._parser(self._handle, **kwargs)
        self._meta, self._fallback = (iter_obj._meta, iter_obj._fallback)

    def __iter__(self):
        """Iterate over BlastXmlIndexer yields qstart_id, start_offset, block's length."""
        qstart_mark = self.qstart_mark
        qend_mark = self.qend_mark
        blast_id_mark = b'Query_'
        block_size = self.block_size
        handle = self._handle
        handle.seek(0)
        re_desc = re.compile(b'<Iteration_query-ID>(.*?)</Iteration_query-ID>\\s+?<Iteration_query-def>(.*?)</Iteration_query-def>')
        re_desc_end = re.compile(b'</Iteration_query-def>')
        counter = 0
        while True:
            start_offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if qstart_mark not in line:
                continue
            assert line.count(qstart_mark) == 1, 'XML without line breaks?'
            assert line.lstrip().startswith(qstart_mark), line
            if qend_mark in line:
                block = line
            else:
                block = [line]
                while line and qend_mark not in line:
                    line = handle.readline()
                    assert qstart_mark not in line, line
                    block.append(line)
                assert line.rstrip().endswith(qend_mark), line
                block = b''.join(block)
            assert block.count(qstart_mark) == 1, 'XML without line breaks? %r' % block
            assert block.count(qend_mark) == 1, 'XML without line breaks? %r' % block
            regx = re.search(re_desc, block)
            try:
                qstart_desc = regx.group(2)
                qstart_id = regx.group(1)
            except AttributeError:
                assert re.search(re_desc_end, block)
                qstart_desc = self._fallback['description'].encode()
                qstart_id = self._fallback['id'].encode()
            if qstart_id.startswith(blast_id_mark):
                qstart_id = qstart_desc.split(b' ', 1)[0]
            yield (qstart_id.decode(), start_offset, len(block))
            counter += 1

    def _parse(self, handle):
        """Overwrite SearchIndexer parse (PRIVATE).

        As we need to set the meta and fallback dictionaries to the parser.
        """
        generator = self._parser(handle, **self._kwargs)
        generator._meta = self._meta
        generator._fallback = self._fallback
        return next(iter(generator))

    def get_raw(self, offset):
        """Return the raw record from the file as a bytes string."""
        qend_mark = self.qend_mark
        handle = self._handle
        handle.seek(offset)
        qresult_raw = handle.readline()
        assert qresult_raw.lstrip().startswith(self.qstart_mark)
        while qend_mark not in qresult_raw:
            qresult_raw += handle.readline()
        assert qresult_raw.rstrip().endswith(qend_mark)
        assert qresult_raw.count(qend_mark) == 1
        return qresult_raw