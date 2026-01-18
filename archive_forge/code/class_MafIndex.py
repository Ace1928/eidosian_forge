import os
from itertools import islice
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequentialAlignmentWriter
class MafIndex:
    """Index for a MAF file.

    The index is a sqlite3 database that is built upon creation of the object
    if necessary, and queried when methods *search* or *get_spliced* are
    used.
    """

    def __init__(self, sqlite_file, maf_file, target_seqname):
        """Indexes or loads the index of a MAF file."""
        if dbapi2 is None:
            from Bio import MissingPythonDependencyError
            raise MissingPythonDependencyError('Python was compiled without the sqlite3 module')
        self._target_seqname = target_seqname
        self._index_filename = sqlite_file
        self._relative_path = os.path.abspath(os.path.dirname(sqlite_file))
        self._maf_file = maf_file
        self._maf_fp = open(self._maf_file)
        if os.path.isfile(sqlite_file):
            self._con = dbapi2.connect(sqlite_file)
            try:
                self._record_count = self.__check_existing_db()
            except ValueError as err:
                self._maf_fp.close()
                self._con.close()
                raise err from None
        else:
            self._con = dbapi2.connect(sqlite_file)
            try:
                self._record_count = self.__make_new_index()
            except ValueError as err:
                self._maf_fp.close()
                self._con.close()
                raise err from None
        self._mafiter = MafIterator(self._maf_fp)

    def close(self):
        """Close the file handle being used to read the data.

        Once called, further use of the index won't work. The sole
        purpose of this method is to allow explicit handle closure
        - for example if you wish to delete the file, on Windows
        you must first close all open handles to that file.
        """
        self._con.close()
        self._record_count = 0

    def __check_existing_db(self):
        """Perform basic sanity checks upon loading an existing index (PRIVATE)."""
        try:
            idx_version = int(self._con.execute("SELECT value FROM meta_data WHERE key = 'version'").fetchone()[0])
            if idx_version != MAFINDEX_VERSION:
                msg = '\n'.join(['Index version (%s) incompatible with this version of MafIndex' % idx_version, 'You might erase the existing index %s for it to be rebuilt.' % self._index_filename])
                raise ValueError(msg)
            filename = self._con.execute("SELECT value FROM meta_data WHERE key = 'filename'").fetchone()[0]
            if os.path.isabs(filename):
                tmp_mafpath = filename
            else:
                tmp_mafpath = os.path.join(self._relative_path, filename.replace('/', os.path.sep))
            if tmp_mafpath != os.path.abspath(self._maf_file):
                raise ValueError(f'Index uses a different file ({filename} != {self._maf_file})')
            db_target = self._con.execute("SELECT value FROM meta_data WHERE key = 'target_seqname'").fetchone()[0]
            if db_target != self._target_seqname:
                raise ValueError('Provided database indexed for %s, expected %s' % (db_target, self._target_seqname))
            record_count = int(self._con.execute("SELECT value FROM meta_data WHERE key = 'record_count'").fetchone()[0])
            if record_count == -1:
                raise ValueError('Unfinished/partial database provided')
            records_found = int(self._con.execute('SELECT COUNT(*) FROM offset_data').fetchone()[0])
            if records_found != record_count:
                raise ValueError('Expected %s records, found %s.  Corrupt index?' % (record_count, records_found))
            return records_found
        except (dbapi2.OperationalError, dbapi2.DatabaseError) as err:
            raise ValueError(f'Problem with SQLite database: {err}') from None

    def __make_new_index(self):
        """Read MAF file and generate SQLite index (PRIVATE)."""
        self._con.execute('CREATE TABLE meta_data (key TEXT, value TEXT);')
        self._con.execute('INSERT INTO meta_data (key, value) VALUES (?, ?);', ('version', MAFINDEX_VERSION))
        self._con.execute("INSERT INTO meta_data (key, value) VALUES ('record_count', -1);")
        self._con.execute('INSERT INTO meta_data (key, value) VALUES (?, ?);', ('target_seqname', self._target_seqname))
        if not os.path.isabs(self._maf_file) and (not os.path.isabs(self._index_filename)):
            mafpath = os.path.relpath(self._maf_file, self._relative_path).replace(os.path.sep, '/')
        elif (os.path.dirname(os.path.abspath(self._maf_file)) + os.path.sep).startswith(self._relative_path + os.path.sep):
            mafpath = os.path.relpath(self._maf_file, self._relative_path).replace(os.path.sep, '/')
        else:
            mafpath = os.path.abspath(self._maf_file)
        self._con.execute('INSERT INTO meta_data (key, value) VALUES (?, ?);', ('filename', mafpath))
        self._con.execute('CREATE TABLE offset_data (bin INTEGER, start INTEGER, end INTEGER, offset INTEGER);')
        insert_count = 0
        mafindex_func = self.__maf_indexer()
        while True:
            batch = list(islice(mafindex_func, 100))
            if not batch:
                break
            self._con.executemany('INSERT INTO offset_data (bin, start, end, offset) VALUES (?,?,?,?);', batch)
            self._con.commit()
            insert_count += len(batch)
        self._con.execute('CREATE INDEX IF NOT EXISTS bin_index ON offset_data(bin);')
        self._con.execute('CREATE INDEX IF NOT EXISTS start_index ON offset_data(start);')
        self._con.execute('CREATE INDEX IF NOT EXISTS end_index ON offset_data(end);')
        self._con.execute(f"UPDATE meta_data SET value = '{insert_count}' WHERE key = 'record_count'")
        self._con.commit()
        return insert_count

    def __maf_indexer(self):
        """Return index information for each bundle (PRIVATE).

        Yields index information for each bundle in the form of
        (bin, start, end, offset) tuples where start and end are
        0-based inclusive coordinates.
        """
        line = self._maf_fp.readline()
        while line:
            if line.startswith('a'):
                offset = self._maf_fp.tell() - len(line)
                while True:
                    line = self._maf_fp.readline()
                    if not line.strip() or line.startswith('a'):
                        raise ValueError('Target for indexing (%s) not found in this bundle' % (self._target_seqname,))
                    elif line.startswith('s'):
                        line_split = line.strip().split()
                        if line_split[1] == self._target_seqname:
                            start = int(line_split[2])
                            size = int(line_split[3])
                            if size != len(line_split[6].replace('-', '')):
                                raise ValueError('Invalid length for target coordinates (expected %s, found %s)' % (size, len(line_split[6].replace('-', ''))))
                            end = start + size - 1
                            yield (self._ucscbin(start, end + 1), start, end, offset)
                            break
            line = self._maf_fp.readline()

    @staticmethod
    def _region2bin(start, end):
        """Find bins that a region may belong to (PRIVATE).

        Converts a region to a list of bins that it may belong to, including largest
        and smallest bins.
        """
        bins = [0, 1]
        bins.extend(range(1 + (start >> 26), 2 + (end - 1 >> 26)))
        bins.extend(range(9 + (start >> 23), 10 + (end - 1 >> 23)))
        bins.extend(range(73 + (start >> 20), 74 + (end - 1 >> 20)))
        bins.extend(range(585 + (start >> 17), 586 + (end - 1 >> 17)))
        return set(bins)

    @staticmethod
    def _ucscbin(start, end):
        """Return the smallest bin a given region will fit into (PRIVATE).

        Adapted from http://genomewiki.ucsc.edu/index.php/Bin_indexing_system
        """
        bin_offsets = [512 + 64 + 8 + 1, 64 + 8 + 1, 8 + 1, 1, 0]
        _bin_first_shift = 17
        _bin_next_shift = 3
        start_bin = start
        end_bin = end - 1
        start_bin >>= _bin_first_shift
        end_bin >>= _bin_first_shift
        for bin_offset in bin_offsets:
            if start_bin == end_bin:
                return bin_offset + start_bin
            start_bin >>= _bin_next_shift
            end_bin >>= _bin_next_shift
        return 0

    def _get_record(self, offset):
        """Retrieve a single MAF record located at the offset provided (PRIVATE)."""
        self._maf_fp.seek(offset)
        return next(self._mafiter)

    def search(self, starts, ends):
        """Search index database for MAF records overlapping ranges provided.

        Returns *MultipleSeqAlignment* results in order by start, then end, then
        internal offset field.

        *starts* should be a list of 0-based start coordinates of segments in the reference.
        *ends* should be the list of the corresponding segment ends
        (in the half-open UCSC convention:
        http://genome.ucsc.edu/blog/the-ucsc-genome-browser-coordinate-counting-systems/).
        """
        if len(starts) != len(ends):
            raise ValueError('Every position in starts must have a match in ends')
        for exonstart, exonend in zip(starts, ends):
            exonlen = exonend - exonstart
            if exonlen < 1:
                raise ValueError('Exon coordinates (%d, %d) invalid: exon length (%d) < 1' % (exonstart, exonend, exonlen))
        con = self._con
        yielded_rec_coords = set()
        for exonstart, exonend in zip(starts, ends):
            try:
                possible_bins = ', '.join(map(str, self._region2bin(exonstart, exonend)))
            except TypeError:
                raise TypeError('Exon coordinates must be integers (start=%d, end=%d)' % (exonstart, exonend)) from None
            result = con.execute('SELECT DISTINCT start, end, offset FROM offset_data WHERE bin IN (%s) AND (end BETWEEN %s AND %s OR %s BETWEEN start AND end) ORDER BY start, end, offset ASC;' % (possible_bins, exonstart, exonend - 1, exonend - 1))
            rows = result.fetchall()
            for rec_start, rec_end, offset in rows:
                if (rec_start, rec_end) in yielded_rec_coords:
                    continue
                else:
                    yielded_rec_coords.add((rec_start, rec_end))
                fetched = self._get_record(int(offset))
                for record in fetched:
                    if record.id == self._target_seqname:
                        start = record.annotations['start']
                        end = start + record.annotations['size'] - 1
                        if not (start == rec_start and end == rec_end):
                            raise ValueError('Expected %s-%s @ offset %s, found %s-%s' % (rec_start, rec_end, offset, start, end))
                yield fetched

    def get_spliced(self, starts, ends, strand=1):
        """Return a multiple alignment of the exact sequence range provided.

        Accepts two lists of start and end positions on target_seqname, representing
        exons to be spliced in silico.  Returns a *MultipleSeqAlignment* of the
        desired sequences spliced together.

        *starts* should be a list of 0-based start coordinates of segments in the reference.
        *ends* should be the list of the corresponding segment ends
        (in the half-open UCSC convention:
        http://genome.ucsc.edu/blog/the-ucsc-genome-browser-coordinate-counting-systems/).

        To ask for the alignment portion corresponding to the first 100
        nucleotides of the reference sequence, you would use
        ``search([0], [100])``
        """
        if strand not in (1, -1):
            raise ValueError(f'Strand must be 1 or -1, got {strand}')
        fetched = list(self.search(starts, ends))
        expected_letters = sum((end - start for start, end in zip(starts, ends)))
        if len(fetched) == 0:
            return MultipleSeqAlignment([SeqRecord(Seq('N' * expected_letters), id=self._target_seqname)])
        all_seqnames = {sequence.id for multiseq in fetched for sequence in multiseq}
        split_by_position = {seq_name: {} for seq_name in all_seqnames}
        total_rec_length = 0
        ref_first_strand = None
        for multiseq in fetched:
            for seqrec in multiseq:
                if seqrec.id == self._target_seqname:
                    try:
                        if ref_first_strand is None:
                            ref_first_strand = seqrec.annotations['strand']
                            if ref_first_strand not in (1, -1):
                                raise ValueError('Strand must be 1 or -1')
                        elif ref_first_strand != seqrec.annotations['strand']:
                            raise ValueError("Encountered strand='%s' on target seqname, expected '%s'" % (seqrec.annotations['strand'], ref_first_strand))
                    except KeyError:
                        raise ValueError('No strand information for target seqname (%s)' % self._target_seqname) from None
                    rec_length = len(seqrec)
                    rec_start = seqrec.annotations['start']
                    ungapped_length = seqrec.annotations['size']
                    rec_end = rec_start + ungapped_length - 1
                    total_rec_length += ungapped_length
                    for seqrec in multiseq:
                        for pos in range(rec_start, rec_end + 1):
                            split_by_position[seqrec.id][pos] = ''
                    break
            else:
                raise ValueError(f'Did not find {self._target_seqname} in alignment bundle')
            real_pos = rec_start
            for gapped_pos in range(rec_length):
                for seqrec in multiseq:
                    if seqrec.id == self._target_seqname:
                        track_val = seqrec.seq[gapped_pos]
                    split_by_position[seqrec.id][real_pos] += seqrec.seq[gapped_pos]
                if track_val != '-' and real_pos < rec_end:
                    real_pos += 1
        if len(split_by_position[self._target_seqname]) != total_rec_length:
            raise ValueError('Target seqname (%s) has %s records, expected %s' % (self._target_seqname, len(split_by_position[self._target_seqname]), total_rec_length))
        realpos_to_len = {pos: len(gapped_fragment) for pos, gapped_fragment in split_by_position[self._target_seqname].items() if len(gapped_fragment) > 1}
        subseq = {}
        for seqid in all_seqnames:
            seq_split = split_by_position[seqid]
            seq_splice = []
            filler_char = 'N' if seqid == self._target_seqname else '-'
            append = seq_splice.append
            for exonstart, exonend in zip(starts, ends):
                for real_pos in range(exonstart, exonend):
                    if real_pos in seq_split:
                        append(seq_split[real_pos])
                    elif real_pos in realpos_to_len:
                        append(filler_char * realpos_to_len[real_pos])
                    else:
                        append(filler_char)
            subseq[seqid] = ''.join(seq_splice)
        if len(subseq[self._target_seqname].replace('-', '')) != expected_letters:
            raise ValueError('Returning %s letters for target seqname (%s), expected %s' % (len(subseq[self._target_seqname].replace('-', '')), self._target_seqname, expected_letters))
        ref_subseq_len = len(subseq[self._target_seqname])
        for seqid, seq in subseq.items():
            if len(seq) != ref_subseq_len:
                raise ValueError('Returning length %s for %s, expected %s' % (len(seq), seqid, ref_subseq_len))
        result_multiseq = []
        for seqid, seq in subseq.items():
            seq = Seq(seq)
            seq = seq if strand == ref_first_strand else seq.reverse_complement()
            result_multiseq.append(SeqRecord(seq, id=seqid, name=seqid, description=''))
        return MultipleSeqAlignment(result_multiseq)

    def __repr__(self):
        """Return a string representation of the index."""
        return 'MafIO.MafIndex(%r, target_seqname=%r)' % (self._maf_fp.name, self._target_seqname)

    def __len__(self):
        """Return the number of records in the index."""
        return self._record_count