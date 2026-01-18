import re
import struct
from Bio import StreamModeError
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
class SffWriter(SequenceWriter):
    """SFF file writer."""

    def __init__(self, target, index=True, xml=None):
        """Initialize an SFF writer object.

        Arguments:
         - target - Output stream opened in binary mode, or a path to a file.
         - index - Boolean argument, should we try and write an index?
         - xml - Optional string argument, xml manifest to be recorded
           in the index block (see function ReadRocheXmlManifest for
           reading this data).

        """
        super().__init__(target, 'wb')
        self._xml = xml
        if index:
            self._index = []
        else:
            self._index = None

    def write_file(self, records):
        """Use this to write an entire file containing the given records."""
        try:
            self._number_of_reads = len(records)
        except TypeError:
            self._number_of_reads = 0
            if not hasattr(self.handle, 'seek') or not hasattr(self.handle, 'tell'):
                raise ValueError('A handle with a seek/tell methods is required in order to record the total record count in the file header (once it is known at the end).') from None
        if self._index is not None and (not (hasattr(self.handle, 'seek') and hasattr(self.handle, 'tell'))):
            import warnings
            warnings.warn('A handle with a seek/tell methods is required in order to record an SFF index.')
            self._index = None
        self._index_start = 0
        self._index_length = 0
        if not hasattr(records, 'next'):
            records = iter(records)
        try:
            record = next(records)
        except StopIteration:
            record = None
        if record is None:
            raise ValueError('Must have at least one sequence')
        try:
            self._key_sequence = record.annotations['flow_key'].encode('ASCII')
            self._flow_chars = record.annotations['flow_chars'].encode('ASCII')
            self._number_of_flows_per_read = len(self._flow_chars)
        except KeyError:
            raise ValueError('Missing SFF flow information') from None
        self.write_header()
        self.write_record(record)
        count = 1
        for record in records:
            self.write_record(record)
            count += 1
        if self._number_of_reads == 0:
            offset = self.handle.tell()
            self.handle.seek(0)
            self._number_of_reads = count
            self.write_header()
            self.handle.seek(offset)
        else:
            assert count == self._number_of_reads
        if self._index is not None:
            self._write_index()
        return count

    def _write_index(self):
        assert len(self._index) == self._number_of_reads
        handle = self.handle
        self._index.sort()
        self._index_start = handle.tell()
        if self._xml is not None:
            xml = self._xml.encode()
        else:
            from Bio import __version__
            xml = f'<!-- This file was output with Biopython {__version__} -->\n'
            xml += '<!-- This XML and index block attempts to mimic Roche SFF files -->\n'
            xml += '<!-- This file may be a combination of multiple SFF files etc -->\n'
            xml = xml.encode()
        xml_len = len(xml)
        fmt = '>I4BLL'
        fmt_size = struct.calcsize(fmt)
        handle.write(_null * fmt_size + xml)
        fmt2 = '>6B'
        assert 6 == struct.calcsize(fmt2)
        self._index.sort()
        index_len = 0
        for name, offset in self._index:
            off3 = offset
            off0 = off3 % 255
            off3 -= off0
            off1 = off3 % 65025
            off3 -= off1
            off2 = off3 % 16581375
            off3 -= off2
            if offset != off0 + off1 + off2 + off3:
                raise RuntimeError('%i -> %i %i %i %i' % (offset, off0, off1, off2, off3))
            off3, off2, off1, off0 = (off3 // 16581375, off2 // 65025, off1 // 255, off0)
            if not (off0 < 255 and off1 < 255 and (off2 < 255) and (off3 < 255)):
                raise RuntimeError('%i -> %i %i %i %i' % (offset, off0, off1, off2, off3))
            handle.write(name + struct.pack(fmt2, 0, off3, off2, off1, off0, 255))
            index_len += len(name) + 6
        self._index_length = fmt_size + xml_len + index_len
        if self._index_length % 8:
            padding = 8 - self._index_length % 8
            handle.write(_null * padding)
        else:
            padding = 0
        offset = handle.tell()
        if offset != self._index_start + self._index_length + padding:
            raise RuntimeError('%i vs %i + %i + %i' % (offset, self._index_start, self._index_length, padding))
        handle.seek(self._index_start)
        handle.write(struct.pack(fmt, 778921588, 49, 46, 48, 48, xml_len, index_len) + xml)
        handle.seek(0)
        self.write_header()
        handle.seek(offset)

    def write_header(self):
        """Write the SFF file header."""
        key_length = len(self._key_sequence)
        fmt = '>I4BQIIHHHB%is%is' % (self._number_of_flows_per_read, key_length)
        if struct.calcsize(fmt) % 8 == 0:
            padding = 0
        else:
            padding = 8 - struct.calcsize(fmt) % 8
        header_length = struct.calcsize(fmt) + padding
        assert header_length % 8 == 0
        header = struct.pack(fmt, 779314790, 0, 0, 0, 1, self._index_start, self._index_length, self._number_of_reads, header_length, key_length, self._number_of_flows_per_read, 1, self._flow_chars, self._key_sequence)
        self.handle.write(header + _null * padding)

    def write_record(self, record):
        """Write a single additional record to the output file.

        This assumes the header has been done.
        """
        name = record.id.encode()
        name_len = len(name)
        seq = bytes(record.seq).upper()
        seq_len = len(seq)
        try:
            quals = record.letter_annotations['phred_quality']
        except KeyError:
            raise ValueError(f'Missing PHRED qualities information for {record.id}') from None
        try:
            flow_values = record.annotations['flow_values']
            flow_index = record.annotations['flow_index']
            if self._key_sequence != record.annotations['flow_key'].encode() or self._flow_chars != record.annotations['flow_chars'].encode():
                raise ValueError('Records have inconsistent SFF flow data')
        except KeyError:
            raise ValueError(f'Missing SFF flow information for {record.id}') from None
        except AttributeError:
            raise ValueError('Header not written yet?') from None
        try:
            clip_qual_left = record.annotations['clip_qual_left']
            if clip_qual_left < 0:
                raise ValueError(f'Negative SFF clip_qual_left value for {record.id}')
            if clip_qual_left:
                clip_qual_left += 1
            clip_qual_right = record.annotations['clip_qual_right']
            if clip_qual_right < 0:
                raise ValueError(f'Negative SFF clip_qual_right value for {record.id}')
            clip_adapter_left = record.annotations['clip_adapter_left']
            if clip_adapter_left < 0:
                raise ValueError(f'Negative SFF clip_adapter_left value for {record.id}')
            if clip_adapter_left:
                clip_adapter_left += 1
            clip_adapter_right = record.annotations['clip_adapter_right']
            if clip_adapter_right < 0:
                raise ValueError(f'Negative SFF clip_adapter_right value for {record.id}')
        except KeyError:
            raise ValueError(f'Missing SFF clipping information for {record.id}') from None
        if self._index is not None:
            offset = self.handle.tell()
            if offset > 4228250624:
                import warnings
                warnings.warn('Read %s has file offset %i, which is too large to store in the Roche SFF index structure. No index block will be recorded.' % (name, offset))
                self._index = None
            else:
                self._index.append((name, self.handle.tell()))
        read_header_fmt = '>2HI4H%is' % name_len
        if struct.calcsize(read_header_fmt) % 8 == 0:
            padding = 0
        else:
            padding = 8 - struct.calcsize(read_header_fmt) % 8
        read_header_length = struct.calcsize(read_header_fmt) + padding
        assert read_header_length % 8 == 0
        data = struct.pack(read_header_fmt, read_header_length, name_len, seq_len, clip_qual_left, clip_qual_right, clip_adapter_left, clip_adapter_right, name) + _null * padding
        assert len(data) == read_header_length
        read_flow_fmt = '>%iH' % self._number_of_flows_per_read
        read_flow_size = struct.calcsize(read_flow_fmt)
        temp_fmt = '>%iB' % seq_len
        data += struct.pack(read_flow_fmt, *flow_values) + struct.pack(temp_fmt, *flow_index) + seq + struct.pack(temp_fmt, *quals)
        padding = (read_flow_size + seq_len * 3) % 8
        if padding:
            padding = 8 - padding
        self.handle.write(data + _null * padding)