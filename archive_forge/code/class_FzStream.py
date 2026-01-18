from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class FzStream(object):
    """
    Wrapper class for struct `fz_stream`.
    fz_stream is a buffered reader capable of seeking in both
    directions.

    Streams are reference counted, so references must be dropped
    by a call to fz_drop_stream.

    Only the data between rp and wp is valid.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_available(self, max):
        """
        Class-aware wrapper for `::fz_available()`.
        	Ask how many bytes are available immediately from
        	a given stream.

        	stm: The stream to read from.

        	max: A hint for the underlying stream; the maximum number of
        	bytes that we are sure we will want to read. If you do not know
        	this number, give 1.

        	Returns the number of bytes immediately available between the
        	read and write pointers. This number is guaranteed only to be 0
        	if we have hit EOF. The number of bytes returned here need have
        	no relation to max (could be larger, could be smaller).
        """
        return _mupdf.FzStream_fz_available(self, max)

    def fz_decomp_image_from_stream(self, image, subarea, indexed, l2factor, l2extra):
        """
        Class-aware wrapper for `::fz_decomp_image_from_stream()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_decomp_image_from_stream(::fz_compressed_image *image, ::fz_irect *subarea, int indexed, int l2factor)` => `(fz_pixmap *, int l2extra)`

        	Decode a subarea of a compressed image. l2factor is the amount
        	of subsampling inbuilt to the stream (i.e. performed by the
        	decoder). If non NULL, l2extra is the extra amount of
        	subsampling that should be performed by this routine. This will
        	be updated on exit to the amount of subsampling that is still
        	required to be done.

        	Returns a kept reference.
        """
        return _mupdf.FzStream_fz_decomp_image_from_stream(self, image, subarea, indexed, l2factor, l2extra)

    def fz_is_cfb_archive(self):
        """
        Class-aware wrapper for `::fz_is_cfb_archive()`.
        	Detect if stream object is a cfb archive.

        	Assumes that the stream object is seekable.
        """
        return _mupdf.FzStream_fz_is_cfb_archive(self)

    def fz_is_eof(self):
        """
        Class-aware wrapper for `::fz_is_eof()`.
        	Query if the stream has reached EOF (during normal bytewise
        	reading).

        	See fz_is_eof_bits for the equivalent function for bitwise
        	reading.
        """
        return _mupdf.FzStream_fz_is_eof(self)

    def fz_is_eof_bits(self):
        """
        Class-aware wrapper for `::fz_is_eof_bits()`.
        	Query if the stream has reached EOF (during bitwise
        	reading).

        	See fz_is_eof for the equivalent function for bytewise
        	reading.
        """
        return _mupdf.FzStream_fz_is_eof_bits(self)

    def fz_is_libarchive_archive(self):
        """
        Class-aware wrapper for `::fz_is_libarchive_archive()`.
        	Detect if stream object is an archive supported by libarchive.

        	Assumes that the stream object is seekable.
        """
        return _mupdf.FzStream_fz_is_libarchive_archive(self)

    def fz_is_tar_archive(self):
        """
        Class-aware wrapper for `::fz_is_tar_archive()`.
        	Detect if stream object is a tar archive.

        	Assumes that the stream object is seekable.
        """
        return _mupdf.FzStream_fz_is_tar_archive(self)

    def fz_is_zip_archive(self):
        """
        Class-aware wrapper for `::fz_is_zip_archive()`.
        	Detect if stream object is a zip archive.

        	Assumes that the stream object is seekable.
        """
        return _mupdf.FzStream_fz_is_zip_archive(self)

    def fz_new_archive_of_size(self, size):
        """ Class-aware wrapper for `::fz_new_archive_of_size()`."""
        return _mupdf.FzStream_fz_new_archive_of_size(self, size)

    def fz_open_a85d(self):
        """
        Class-aware wrapper for `::fz_open_a85d()`.
        	a85d filter performs ASCII 85 Decoding of data read
        	from the chained filter.
        """
        return _mupdf.FzStream_fz_open_a85d(self)

    def fz_open_aesd(self, key, keylen):
        """
        Class-aware wrapper for `::fz_open_aesd()`.
        	aesd filter performs AES decoding of data read from the chained
        	filter using the supplied key.
        """
        return _mupdf.FzStream_fz_open_aesd(self, key, keylen)

    def fz_open_ahxd(self):
        """
        Class-aware wrapper for `::fz_open_ahxd()`.
        	ahxd filter performs ASCII Hex decoding of data read
        	from the chained filter.
        """
        return _mupdf.FzStream_fz_open_ahxd(self)

    def fz_open_arc4(self, key, keylen):
        """
        Class-aware wrapper for `::fz_open_arc4()`.
        	arc4 filter performs RC4 decoding of data read from the chained
        	filter using the supplied key.
        """
        return _mupdf.FzStream_fz_open_arc4(self, key, keylen)

    def fz_open_archive_with_stream(self):
        """
        Class-aware wrapper for `::fz_open_archive_with_stream()`.
        	Open zip or tar archive stream.

        	Open an archive using a seekable stream object rather than
        	opening a file or directory on disk.
        """
        return _mupdf.FzStream_fz_open_archive_with_stream(self)

    def fz_open_cfb_archive_with_stream(self):
        """
        Class-aware wrapper for `::fz_open_cfb_archive_with_stream()`.
        	Open a cfb file as an archive.

        	Open an archive using a seekable stream object rather than
        	opening a file or directory on disk.

        	An exception is thrown if the file is not recognised as a chm.
        """
        return _mupdf.FzStream_fz_open_cfb_archive_with_stream(self)

    def fz_open_dctd(self, color_transform, invert_cmyk, l2factor, jpegtables):
        """
        Class-aware wrapper for `::fz_open_dctd()`.
        	dctd filter performs DCT (JPEG) decoding of data read
        	from the chained filter.

        	color_transform implements the PDF color_transform option
        		use -1 for default behavior
        		use 0 to disable YUV-RGB / YCCK-CMYK transforms
        		use 1 to enable YUV-RGB / YCCK-CMYK transforms

        	invert_cmyk implements the necessary inversion for Photoshop CMYK images
        		use 0 if embedded in PDF
        		use 1 if not embedded in PDF

        	For subsampling on decode, set l2factor to the log2 of the
        	reduction required (therefore 0 = full size decode).

        	jpegtables is an optional stream from which the JPEG tables
        	can be read. Use NULL if not required.
        """
        return _mupdf.FzStream_fz_open_dctd(self, color_transform, invert_cmyk, l2factor, jpegtables)

    def fz_open_endstream_filter(self, len, offset):
        """
        Class-aware wrapper for `::fz_open_endstream_filter()`.
        	The endstream filter reads a PDF substream, and starts to look
        	for an 'endstream' token after the specified length.
        """
        return _mupdf.FzStream_fz_open_endstream_filter(self, len, offset)

    def fz_open_faxd(self, k, end_of_line, encoded_byte_align, columns, rows, end_of_block, black_is_1):
        """
        Class-aware wrapper for `::fz_open_faxd()`.
        	faxd filter performs FAX decoding of data read from
        	the chained filter.

        	k: see fax specification (fax default is 0).

        	end_of_line: whether we expect end of line markers (fax default
        	is 0).

        	encoded_byte_align: whether we align to bytes after each line
        	(fax default is 0).

        	columns: how many columns in the image (fax default is 1728).

        	rows: 0 for unspecified or the number of rows of data to expect.

        	end_of_block: whether we expect end of block markers (fax
        	default is 1).

        	black_is_1: determines the polarity of the image (fax default is
        	0).
        """
        return _mupdf.FzStream_fz_open_faxd(self, k, end_of_line, encoded_byte_align, columns, rows, end_of_block, black_is_1)

    def fz_open_flated(self, window_bits):
        """
        Class-aware wrapper for `::fz_open_flated()`.
        	flated filter performs LZ77 decoding (inflating) of data read
        	from the chained filter.

        	window_bits: How large a decompression window to use. Typically
        	15. A negative number, -n, means to use n bits, but to expect
        	raw data with no header.
        """
        return _mupdf.FzStream_fz_open_flated(self, window_bits)

    def fz_open_image_decomp_stream(self, arg_1, l2factor):
        """
        Class-aware wrapper for `::fz_open_image_decomp_stream()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_open_image_decomp_stream(::fz_compression_params *arg_1)` => `(fz_stream *, int l2factor)`

        	Open a stream to read the decompressed version of another stream
        	with optional log2 subsampling.
        """
        return _mupdf.FzStream_fz_open_image_decomp_stream(self, arg_1, l2factor)

    def fz_open_jbig2d(self, globals, embedded):
        """
        Class-aware wrapper for `::fz_open_jbig2d()`.
        	Open a filter that performs jbig2 decompression on the chained
        	stream, using the optional globals record.
        """
        return _mupdf.FzStream_fz_open_jbig2d(self, globals, embedded)

    def fz_open_leecher(self, buf):
        """
        Class-aware wrapper for `::fz_open_leecher()`.
        	Attach a filter to a stream that will store any
        	characters read from the stream into the supplied buffer.

        	chain: The underlying stream to leech from.

        	buf: The buffer into which the read data should be appended.
        	The buffer will be resized as required.

        	Returns pointer to newly created stream. May throw exceptions on
        	failure to allocate.
        """
        return _mupdf.FzStream_fz_open_leecher(self, buf)

    def fz_open_libarchive_archive_with_stream(self):
        """
        Class-aware wrapper for `::fz_open_libarchive_archive_with_stream()`.
        	Open an archive using libarchive.

        	Open an archive using a seekable stream object rather than
        	opening a file or directory on disk.

        	An exception is thrown if the stream is not supported by libarchive.
        """
        return _mupdf.FzStream_fz_open_libarchive_archive_with_stream(self)

    def fz_open_libarchived(self):
        """
        Class-aware wrapper for `::fz_open_libarchived()`.
        	libarchived filter performs generic compressed decoding of data
        	in any format understood by libarchive from the chained filter.

        	This will throw an exception if libarchive is not built in, or
        	if the compression format is not recognised.
        """
        return _mupdf.FzStream_fz_open_libarchived(self)

    def fz_open_lzwd(self, early_change, min_bits, reverse_bits, old_tiff):
        """
        Class-aware wrapper for `::fz_open_lzwd()`.
        	lzwd filter performs LZW decoding of data read from the chained
        	filter.

        	early_change: (Default 1) specifies whether to change codes 1
        	bit early.

        	min_bits: (Default 9) specifies the minimum number of bits to
        	use.

        	reverse_bits: (Default 0) allows for compatibility with gif and
        	old style tiffs (1).

        	old_tiff: (Default 0) allows for different handling of the clear
        	code, as found in old style tiffs.
        """
        return _mupdf.FzStream_fz_open_lzwd(self, early_change, min_bits, reverse_bits, old_tiff)

    def fz_open_null_filter(self, len, offset):
        """
        Class-aware wrapper for `::fz_open_null_filter()`.
        	The null filter reads a specified amount of data from the
        	substream.
        """
        return _mupdf.FzStream_fz_open_null_filter(self, len, offset)

    def fz_open_predict(self, predictor, columns, colors, bpc):
        """
        Class-aware wrapper for `::fz_open_predict()`.
        	predict filter performs pixel prediction on data read from
        	the chained filter.

        	predictor: 1 = copy, 2 = tiff, other = inline PNG predictor

        	columns: width of image in pixels

        	colors: number of components.

        	bpc: bits per component (typically 8)
        """
        return _mupdf.FzStream_fz_open_predict(self, predictor, columns, colors, bpc)

    def fz_open_range_filter(self, ranges, nranges):
        """
        Class-aware wrapper for `::fz_open_range_filter()`.
        	The range filter copies data from specified ranges of the
        	chained stream.
        """
        return _mupdf.FzStream_fz_open_range_filter(self, ranges, nranges)

    def fz_open_rld(self):
        """
        Class-aware wrapper for `::fz_open_rld()`.
        	rld filter performs Run Length Decoding of data read
        	from the chained filter.
        """
        return _mupdf.FzStream_fz_open_rld(self)

    def fz_open_sgilog16(self, w):
        """
        Class-aware wrapper for `::fz_open_sgilog16()`.
        	SGI Log 16bit (greyscale) decode from the chained filter.
        	Decodes lines of w pixels to 8bpp greyscale.
        """
        return _mupdf.FzStream_fz_open_sgilog16(self, w)

    def fz_open_sgilog24(self, w):
        """
        Class-aware wrapper for `::fz_open_sgilog24()`.
        	SGI Log 24bit (LUV) decode from the chained filter.
        	Decodes lines of w pixels to 8bpc rgb.
        """
        return _mupdf.FzStream_fz_open_sgilog24(self, w)

    def fz_open_sgilog32(self, w):
        """
        Class-aware wrapper for `::fz_open_sgilog32()`.
        	SGI Log 32bit (LUV) decode from the chained filter.
        	Decodes lines of w pixels to 8bpc rgb.
        """
        return _mupdf.FzStream_fz_open_sgilog32(self, w)

    def fz_open_tar_archive_with_stream(self):
        """
        Class-aware wrapper for `::fz_open_tar_archive_with_stream()`.
        	Open a tar archive stream.

        	Open an archive using a seekable stream object rather than
        	opening a file or directory on disk.

        	An exception is thrown if the stream is not a tar archive as
        	indicated by the presence of a tar signature.

        """
        return _mupdf.FzStream_fz_open_tar_archive_with_stream(self)

    def fz_open_thunder(self, w):
        """
        Class-aware wrapper for `::fz_open_thunder()`.
        	4bit greyscale Thunderscan decoding from the chained filter.
        	Decodes lines of w pixels to 8bpp greyscale.
        """
        return _mupdf.FzStream_fz_open_thunder(self, w)

    def fz_open_zip_archive_with_stream(self):
        """
        Class-aware wrapper for `::fz_open_zip_archive_with_stream()`.
        	Open a zip archive stream.

        	Open an archive using a seekable stream object rather than
        	opening a file or directory on disk.

        	An exception is thrown if the stream is not a zip archive as
        	indicated by the presence of a zip signature.

        """
        return _mupdf.FzStream_fz_open_zip_archive_with_stream(self)

    def fz_parse_xml_stream(self, preserve_white):
        """
        Class-aware wrapper for `::fz_parse_xml_stream()`.
        	Parse the contents of buffer into a tree of xml nodes.

        	preserve_white: whether to keep or delete all-whitespace nodes.
        """
        return _mupdf.FzStream_fz_parse_xml_stream(self, preserve_white)

    def fz_peek_byte(self):
        """
        Class-aware wrapper for `::fz_peek_byte()`.
        	Peek at the next byte in a stream.

        	stm: The stream to peek at.

        	Returns -1 for EOF, or the next byte that will be read.
        """
        return _mupdf.FzStream_fz_peek_byte(self)

    def fz_read(self, data, len):
        """
        Class-aware wrapper for `::fz_read()`.
        	Read from a stream into a given data block.

        	stm: The stream to read from.

        	data: The data block to read into.

        	len: The length of the data block (in bytes).

        	Returns the number of bytes read. May throw exceptions.
        """
        return _mupdf.FzStream_fz_read(self, data, len)

    def fz_read_all(self, initial):
        """
        Class-aware wrapper for `::fz_read_all()`.
        	Read all of a stream into a buffer.

        	stm: The stream to read from

        	initial: Suggested initial size for the buffer.

        	Returns a buffer created from reading from the stream. May throw
        	exceptions on failure to allocate.
        """
        return _mupdf.FzStream_fz_read_all(self, initial)

    def fz_read_best(self, initial, truncated, worst_case):
        """
        Class-aware wrapper for `::fz_read_best()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_read_best(size_t initial, size_t worst_case)` => `(fz_buffer *, int truncated)`

        	Attempt to read a stream into a buffer. If truncated
        	is NULL behaves as fz_read_all, sets a truncated flag in case of
        	error.

        	stm: The stream to read from.

        	initial: Suggested initial size for the buffer.

        	truncated: Flag to store success/failure indication in.

        	worst_case: 0 for unknown, otherwise an upper bound for the
        	size of the stream.

        	Returns a buffer created from reading from the stream.
        """
        return _mupdf.FzStream_fz_read_best(self, initial, truncated, worst_case)

    def fz_read_bits(self, n):
        """
        Class-aware wrapper for `::fz_read_bits()`.
        	Read the next n bits from a stream (assumed to
        	be packed most significant bit first).

        	stm: The stream to read from.

        	n: The number of bits to read, between 1 and 8*sizeof(int)
        	inclusive.

        	Returns -1 for EOF, or the required number of bits.
        """
        return _mupdf.FzStream_fz_read_bits(self, n)

    def fz_read_byte(self):
        """
        Class-aware wrapper for `::fz_read_byte()`.
        	Read the next byte from a stream.

        	stm: The stream t read from.

        	Returns -1 for end of stream, or the next byte. May
        	throw exceptions.
        """
        return _mupdf.FzStream_fz_read_byte(self)

    def fz_read_float(self):
        """ Class-aware wrapper for `::fz_read_float()`."""
        return _mupdf.FzStream_fz_read_float(self)

    def fz_read_float_le(self):
        """ Class-aware wrapper for `::fz_read_float_le()`."""
        return _mupdf.FzStream_fz_read_float_le(self)

    def fz_read_int16(self):
        """ Class-aware wrapper for `::fz_read_int16()`."""
        return _mupdf.FzStream_fz_read_int16(self)

    def fz_read_int16_le(self):
        """ Class-aware wrapper for `::fz_read_int16_le()`."""
        return _mupdf.FzStream_fz_read_int16_le(self)

    def fz_read_int32(self):
        """ Class-aware wrapper for `::fz_read_int32()`."""
        return _mupdf.FzStream_fz_read_int32(self)

    def fz_read_int32_le(self):
        """ Class-aware wrapper for `::fz_read_int32_le()`."""
        return _mupdf.FzStream_fz_read_int32_le(self)

    def fz_read_int64(self):
        """ Class-aware wrapper for `::fz_read_int64()`."""
        return _mupdf.FzStream_fz_read_int64(self)

    def fz_read_int64_le(self):
        """ Class-aware wrapper for `::fz_read_int64_le()`."""
        return _mupdf.FzStream_fz_read_int64_le(self)

    def fz_read_line(self, buf, max):
        """
        Class-aware wrapper for `::fz_read_line()`.
        	Read a line from stream into the buffer until either a
        	terminating newline or EOF, which it replaces with a null byte
        	('').

        	Returns buf on success, and NULL when end of file occurs while
        	no characters have been read.
        """
        return _mupdf.FzStream_fz_read_line(self, buf, max)

    def fz_read_rbits(self, n):
        """
        Class-aware wrapper for `::fz_read_rbits()`.
        	Read the next n bits from a stream (assumed to
        	be packed least significant bit first).

        	stm: The stream to read from.

        	n: The number of bits to read, between 1 and 8*sizeof(int)
        	inclusive.

        	Returns (unsigned int)-1 for EOF, or the required number of bits.
        """
        return _mupdf.FzStream_fz_read_rbits(self, n)

    def fz_read_rune(self):
        """
        Class-aware wrapper for `::fz_read_rune()`.
        	Read a utf-8 rune from a stream.

        	In the event of encountering badly formatted utf-8 codes
        	(such as a leading code with an unexpected number of following
        	codes) no error/exception is given, but undefined values may be
        	returned.
        """
        return _mupdf.FzStream_fz_read_rune(self)

    def fz_read_string(self, buffer, len):
        """
        Class-aware wrapper for `::fz_read_string()`.
        	Read a null terminated string from the stream into
        	a buffer of a given length. The buffer will be null terminated.
        	Throws on failure (including the failure to fit the entire
        	string including the terminator into the buffer).
        """
        return _mupdf.FzStream_fz_read_string(self, buffer, len)

    def fz_read_uint16(self):
        """
        Class-aware wrapper for `::fz_read_uint16()`.
        	fz_read_[u]int(16|24|32|64)(_le)?

        	Read a 16/32/64 bit signed/unsigned integer from stream,
        	in big or little-endian byte orders.

        	Throws an exception if EOF is encountered.
        """
        return _mupdf.FzStream_fz_read_uint16(self)

    def fz_read_uint16_le(self):
        """ Class-aware wrapper for `::fz_read_uint16_le()`."""
        return _mupdf.FzStream_fz_read_uint16_le(self)

    def fz_read_uint24(self):
        """ Class-aware wrapper for `::fz_read_uint24()`."""
        return _mupdf.FzStream_fz_read_uint24(self)

    def fz_read_uint24_le(self):
        """ Class-aware wrapper for `::fz_read_uint24_le()`."""
        return _mupdf.FzStream_fz_read_uint24_le(self)

    def fz_read_uint32(self):
        """ Class-aware wrapper for `::fz_read_uint32()`."""
        return _mupdf.FzStream_fz_read_uint32(self)

    def fz_read_uint32_le(self):
        """ Class-aware wrapper for `::fz_read_uint32_le()`."""
        return _mupdf.FzStream_fz_read_uint32_le(self)

    def fz_read_uint64(self):
        """ Class-aware wrapper for `::fz_read_uint64()`."""
        return _mupdf.FzStream_fz_read_uint64(self)

    def fz_read_uint64_le(self):
        """ Class-aware wrapper for `::fz_read_uint64_le()`."""
        return _mupdf.FzStream_fz_read_uint64_le(self)

    def fz_read_utf16_be(self):
        """ Class-aware wrapper for `::fz_read_utf16_be()`."""
        return _mupdf.FzStream_fz_read_utf16_be(self)

    def fz_read_utf16_le(self):
        """
        Class-aware wrapper for `::fz_read_utf16_le()`.
        	Read a utf-16 rune from a stream. (little endian and
        	big endian respectively).

        	In the event of encountering badly formatted utf-16 codes
        	(mismatched surrogates) no error/exception is given, but
        	undefined values may be returned.
        """
        return _mupdf.FzStream_fz_read_utf16_le(self)

    def fz_seek(self, offset, whence):
        """
        Class-aware wrapper for `::fz_seek()`.
        	Seek within a stream.

        	stm: The stream to seek within.

        	offset: The offset to seek to.

        	whence: From where the offset is measured (see fseek).
        	SEEK_SET - start of stream.
        	SEEK_CUR - current position.
        	SEEK_END - end of stream.

        """
        return _mupdf.FzStream_fz_seek(self, offset, whence)

    def fz_skip(self, len):
        """
        Class-aware wrapper for `::fz_skip()`.
        	Read from a stream discarding data.

        	stm: The stream to read from.

        	len: The number of bytes to read.

        	Returns the number of bytes read. May throw exceptions.
        """
        return _mupdf.FzStream_fz_skip(self, len)

    def fz_skip_space(self):
        """
        Class-aware wrapper for `::fz_skip_space()`.
        	Skip over whitespace (bytes <= 32) in a stream.
        """
        return _mupdf.FzStream_fz_skip_space(self)

    def fz_skip_string(self, str):
        """
        Class-aware wrapper for `::fz_skip_string()`.
        	Skip over a given string in a stream. Return 0 if successfully
        	skipped, non-zero otherwise. As many characters will be skipped
        	over as matched in the string.
        """
        return _mupdf.FzStream_fz_skip_string(self, str)

    def fz_sync_bits(self):
        """
        Class-aware wrapper for `::fz_sync_bits()`.
        	Called after reading bits to tell the stream
        	that we are about to return to reading bytewise. Resyncs
        	the stream to whole byte boundaries.
        """
        return _mupdf.FzStream_fz_sync_bits(self)

    def fz_tell(self):
        """
        Class-aware wrapper for `::fz_tell()`.
        	return the current reading position within a stream
        """
        return _mupdf.FzStream_fz_tell(self)

    def fz_try_open_archive_with_stream(self):
        """
        Class-aware wrapper for `::fz_try_open_archive_with_stream()`.
        	Open zip or tar archive stream.

        	Does the same as fz_open_archive_with_stream, but will not throw
        	an error in the event of failing to recognise the format. Will
        	still throw errors in other cases though!
        """
        return _mupdf.FzStream_fz_try_open_archive_with_stream(self)

    def fz_unpack_stream(self, depth, w, h, n, indexed, pad, skip):
        """ Class-aware wrapper for `::fz_unpack_stream()`."""
        return _mupdf.FzStream_fz_unpack_stream(self, depth, w, h, n, indexed, pad, skip)

    def fz_unread_byte(self):
        """
        Class-aware wrapper for `::fz_unread_byte()`.
        	Unread the single last byte successfully
        	read from a stream. Do not call this without having
        	successfully read a byte.

        	stm: The stream to operate upon.
        """
        return _mupdf.FzStream_fz_unread_byte(self)

    def pdf_load_cmap(self):
        """ Class-aware wrapper for `::pdf_load_cmap()`."""
        return _mupdf.FzStream_pdf_load_cmap(self)

    def pdf_open_crypt(self, crypt, num, gen):
        """ Class-aware wrapper for `::pdf_open_crypt()`."""
        return _mupdf.FzStream_pdf_open_crypt(self, crypt, num, gen)

    def pdf_open_crypt_with_filter(self, crypt, name, num, gen):
        """ Class-aware wrapper for `::pdf_open_crypt_with_filter()`."""
        return _mupdf.FzStream_pdf_open_crypt_with_filter(self, crypt, name, num, gen)

    def pdf_open_document_with_stream(self):
        """ Class-aware wrapper for `::pdf_open_document_with_stream()`."""
        return _mupdf.FzStream_pdf_open_document_with_stream(self)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_stream()`.
        		Create a new stream object with the given
        		internal state and function pointers.

        		state: Internal state (opaque to everything but implementation).

        		next: Should provide the next set of bytes (up to max) of stream
        		data. Return the number of bytes read, or EOF when there is no
        		more data.

        		drop: Should clean up and free the internal state. May not
        		throw exceptions.


        |

        *Overload 2:*
         Constructor using `fz_open_file()`.
        		Open the named file and wrap it in a stream.

        		filename: Path to a file. On non-Windows machines the filename
        		should be exactly as it would be passed to fopen(2). On Windows
        		machines, the path should be UTF-8 encoded so that non-ASCII
        		characters can be represented. Other platforms do the encoding
        		as standard anyway (and in most cases, particularly for MacOS
        		and Linux, the encoding they use is UTF-8 anyway).


        |

        *Overload 3:*
         Constructor using `fz_open_file_ptr_no_close()`.
        		Create a stream from a FILE * that will not be closed
        		when the stream is dropped.


        |

        *Overload 4:*
         Constructor using `fz_open_memory()`.
        		Open a block of memory as a stream.

        		data: Pointer to start of data block. Ownership of the data
        		block is NOT passed in.

        		len: Number of bytes in data block.

        		Returns pointer to newly created stream. May throw exceptions on
        		failure to allocate.


        |

        *Overload 5:*
         Construct using fz_open_file().

        |

        *Overload 6:*
         Copy constructor using `fz_keep_stream()`.

        |

        *Overload 7:*
         Default constructor, sets `m_internal` to null.

        |

        *Overload 8:*
         Constructor using raw copy of pre-existing `::fz_stream`.
        """
        _mupdf.FzStream_swiginit(self, _mupdf.new_FzStream(*args))
    __swig_destroy__ = _mupdf.delete_FzStream

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzStream_m_internal_value(self)
    m_internal = property(_mupdf.FzStream_m_internal_get, _mupdf.FzStream_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzStream_s_num_instances_get, _mupdf.FzStream_s_num_instances_set)