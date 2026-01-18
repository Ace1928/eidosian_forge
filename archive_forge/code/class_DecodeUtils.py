from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
class DecodeUtils:
    """Unpacking utility functions used in the decoder.

    All methods on this class take raw bytes and return  a tuple containing
    the value parsed from the bytes and the number of bytes consumed to parse
    that value.
    """
    UINT8_BYTE_FORMAT = '!B'
    UINT16_BYTE_FORMAT = '!H'
    UINT32_BYTE_FORMAT = '!I'
    INT8_BYTE_FORMAT = '!b'
    INT16_BYTE_FORMAT = '!h'
    INT32_BYTE_FORMAT = '!i'
    INT64_BYTE_FORMAT = '!q'
    PRELUDE_BYTE_FORMAT = '!III'
    UINT_BYTE_FORMAT = {1: UINT8_BYTE_FORMAT, 2: UINT16_BYTE_FORMAT, 4: UINT32_BYTE_FORMAT}

    @staticmethod
    def unpack_true(data):
        """This method consumes none of the provided bytes and returns True.

        :type data: bytes
        :param data: The bytes to parse from. This is ignored in this method.

        :rtype: tuple
        :rtype: (bool, int)
        :returns: The tuple (True, 0)
        """
        return (True, 0)

    @staticmethod
    def unpack_false(data):
        """This method consumes none of the provided bytes and returns False.

        :type data: bytes
        :param data: The bytes to parse from. This is ignored in this method.

        :rtype: tuple
        :rtype: (bool, int)
        :returns: The tuple (False, 0)
        """
        return (False, 0)

    @staticmethod
    def unpack_uint8(data):
        """Parse an unsigned 8-bit integer from the bytes.

        :type data: bytes
        :param data: The bytes to parse from.

        :rtype: (int, int)
        :returns: A tuple containing the (parsed integer value, bytes consumed)
        """
        value = unpack(DecodeUtils.UINT8_BYTE_FORMAT, data[:1])[0]
        return (value, 1)

    @staticmethod
    def unpack_uint32(data):
        """Parse an unsigned 32-bit integer from the bytes.

        :type data: bytes
        :param data: The bytes to parse from.

        :rtype: (int, int)
        :returns: A tuple containing the (parsed integer value, bytes consumed)
        """
        value = unpack(DecodeUtils.UINT32_BYTE_FORMAT, data[:4])[0]
        return (value, 4)

    @staticmethod
    def unpack_int8(data):
        """Parse a signed 8-bit integer from the bytes.

        :type data: bytes
        :param data: The bytes to parse from.

        :rtype: (int, int)
        :returns: A tuple containing the (parsed integer value, bytes consumed)
        """
        value = unpack(DecodeUtils.INT8_BYTE_FORMAT, data[:1])[0]
        return (value, 1)

    @staticmethod
    def unpack_int16(data):
        """Parse a signed 16-bit integer from the bytes.

        :type data: bytes
        :param data: The bytes to parse from.

        :rtype: tuple
        :rtype: (int, int)
        :returns: A tuple containing the (parsed integer value, bytes consumed)
        """
        value = unpack(DecodeUtils.INT16_BYTE_FORMAT, data[:2])[0]
        return (value, 2)

    @staticmethod
    def unpack_int32(data):
        """Parse a signed 32-bit integer from the bytes.

        :type data: bytes
        :param data: The bytes to parse from.

        :rtype: tuple
        :rtype: (int, int)
        :returns: A tuple containing the (parsed integer value, bytes consumed)
        """
        value = unpack(DecodeUtils.INT32_BYTE_FORMAT, data[:4])[0]
        return (value, 4)

    @staticmethod
    def unpack_int64(data):
        """Parse a signed 64-bit integer from the bytes.

        :type data: bytes
        :param data: The bytes to parse from.

        :rtype: tuple
        :rtype: (int, int)
        :returns: A tuple containing the (parsed integer value, bytes consumed)
        """
        value = unpack(DecodeUtils.INT64_BYTE_FORMAT, data[:8])[0]
        return (value, 8)

    @staticmethod
    def unpack_byte_array(data, length_byte_size=2):
        """Parse a variable length byte array from the bytes.

        The bytes are expected to be in the following format:
            [ length ][0 ... length bytes]
        where length is an unsigned integer represented in the smallest number
        of bytes to hold the maximum length of the array.

        :type data: bytes
        :param data: The bytes to parse from.

        :type length_byte_size: int
        :param length_byte_size: The byte size of the preceding integer that
        represents the length of the array. Supported values are 1, 2, and 4.

        :rtype: (bytes, int)
        :returns: A tuple containing the (parsed byte array, bytes consumed).
        """
        uint_byte_format = DecodeUtils.UINT_BYTE_FORMAT[length_byte_size]
        length = unpack(uint_byte_format, data[:length_byte_size])[0]
        bytes_end = length + length_byte_size
        array_bytes = data[length_byte_size:bytes_end]
        return (array_bytes, bytes_end)

    @staticmethod
    def unpack_utf8_string(data, length_byte_size=2):
        """Parse a variable length utf-8 string from the bytes.

        The bytes are expected to be in the following format:
            [ length ][0 ... length bytes]
        where length is an unsigned integer represented in the smallest number
        of bytes to hold the maximum length of the array and the following
        bytes are a valid utf-8 string.

        :type data: bytes
        :param bytes: The bytes to parse from.

        :type length_byte_size: int
        :param length_byte_size: The byte size of the preceding integer that
        represents the length of the array. Supported values are 1, 2, and 4.

        :rtype: (str, int)
        :returns: A tuple containing the (utf-8 string, bytes consumed).
        """
        array_bytes, consumed = DecodeUtils.unpack_byte_array(data, length_byte_size)
        return (array_bytes.decode('utf-8'), consumed)

    @staticmethod
    def unpack_uuid(data):
        """Parse a 16-byte uuid from the bytes.

        :type data: bytes
        :param data: The bytes to parse from.

        :rtype: (bytes, int)
        :returns: A tuple containing the (uuid bytes, bytes consumed).
        """
        return (data[:16], 16)

    @staticmethod
    def unpack_prelude(data):
        """Parse the prelude for an event stream message from the bytes.

        The prelude for an event stream message has the following format:
            [total_length][header_length][prelude_crc]
        where each field is an unsigned 32-bit integer.

        :rtype: ((int, int, int), int)
        :returns: A tuple of ((total_length, headers_length, prelude_crc),
        consumed)
        """
        return (unpack(DecodeUtils.PRELUDE_BYTE_FORMAT, data), _PRELUDE_LENGTH)