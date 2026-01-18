import struct
import sys
class Crc:
    """Compute a Cyclic Redundancy Check (CRC) using the specified polynomial.

    Instances of this class have the same interface as the md5 and sha modules
    in the Python standard library.  See the documentation for these modules
    for examples of how to use a Crc instance.

    The string representation of a Crc instance identifies the polynomial,
    initial value, XOR out value, and the current CRC value.  The print
    statement can be used to output this information.

    If you need to generate a C/C++ function for use in another application,
    use the generateCode method.  If you need to generate code for another
    language, subclass Crc and override the generateCode method.

    The following are the parameters supplied to the constructor.

    poly -- The generator polynomial to use in calculating the CRC.  The value
    is specified as a Python integer or long integer.  The bits in this integer
    are the coefficients of the polynomial.  The only polynomials allowed are
    those that generate 8, 16, 24, 32, or 64 bit CRCs.

    initCrc -- Initial value used to start the CRC calculation.  This initial
    value should be the initial shift register value XORed with the final XOR
    value.  That is equivalent to the CRC result the algorithm should return for
    a zero-length string.  Defaults to all bits set because that starting value
    will take leading zero bytes into account.  Starting with zero will ignore
    all leading zero bytes.

    rev -- A flag that selects a bit reversed algorithm when True.  Defaults to
    True because the bit reversed algorithms are more efficient.

    xorOut -- Final value to XOR with the calculated CRC value.  Used by some
    CRC algorithms.  Defaults to zero.
    """

    def __init__(self, poly, initCrc=_CRC_INIT, rev=True, xorOut=0, initialize=True):
        if not initialize:
            return
        sizeBits, initCrc, xorOut = _verifyParams(poly, initCrc, xorOut)
        self.digest_size = sizeBits // 8
        self.initCrc = initCrc
        self.xorOut = xorOut
        self.poly = poly
        self.reverse = rev
        crcfun, table = _mkCrcFun(poly, sizeBits, initCrc, rev, xorOut)
        self._crc = crcfun
        self.table = table
        self.crcValue = self.initCrc

    def __str__(self):
        lst = []
        lst.append('poly = 0x%X' % self.poly)
        lst.append('reverse = %s' % self.reverse)
        fmt = '0x%%0%dX' % (self.digest_size * 2)
        lst.append('initCrc  = %s' % (fmt % self.initCrc))
        lst.append('xorOut   = %s' % (fmt % self.xorOut))
        lst.append('crcValue = %s' % (fmt % self.crcValue))
        return '\n'.join(lst)

    def new(self, arg=None):
        """Create a new instance of the Crc class initialized to the same
        values as the original instance.  The current CRC is set to the initial
        value.  If a string is provided in the optional arg parameter, it is
        passed to the update method.
        """
        n = Crc(poly=None, initialize=False)
        n._crc = self._crc
        n.digest_size = self.digest_size
        n.initCrc = self.initCrc
        n.xorOut = self.xorOut
        n.table = self.table
        n.crcValue = self.initCrc
        n.reverse = self.reverse
        n.poly = self.poly
        if arg is not None:
            n.update(arg)
        return n

    def copy(self):
        """Create a new instance of the Crc class initialized to the same
        values as the original instance.  The current CRC is set to the current
        value.  This allows multiple CRC calculations using a common initial
        string.
        """
        c = self.new()
        c.crcValue = self.crcValue
        return c

    def update(self, data):
        """Update the current CRC value using the string specified as the data
        parameter.
        """
        self.crcValue = self._crc(data, self.crcValue)

    def digest(self):
        """Return the current CRC value as a string of bytes.  The length of
        this string is specified in the digest_size attribute.
        """
        n = self.digest_size
        crc = self.crcValue
        lst = []
        while n > 0:
            lst.append(chr(crc & 255))
            crc = crc >> 8
            n -= 1
        lst.reverse()
        return ''.join(lst)

    def hexdigest(self):
        """Return the current CRC value as a string of hex digits.  The length
        of this string is twice the digest_size attribute.
        """
        n = self.digest_size
        crc = self.crcValue
        lst = []
        while n > 0:
            lst.append('%02X' % (crc & 255))
            crc = crc >> 8
            n -= 1
        lst.reverse()
        return ''.join(lst)

    def generateCode(self, functionName, out, dataType=None, crcType=None):
        """Generate a C/C++ function.

        functionName -- String specifying the name of the function.

        out -- An open file-like object with a write method.  This specifies
        where the generated code is written.

        dataType -- An optional parameter specifying the data type of the input
        data to the function.  Defaults to UINT8.

        crcType -- An optional parameter specifying the data type of the CRC
        value.  Defaults to one of UINT8, UINT16, UINT32, or UINT64 depending
        on the size of the CRC value.
        """
        if dataType is None:
            dataType = 'UINT8'
        if crcType is None:
            size = 8 * self.digest_size
            if size == 24:
                size = 32
            crcType = 'UINT%d' % size
        if self.digest_size == 1:
            crcAlgor = 'table[*data ^ (%s)crc]'
        elif self.reverse:
            crcAlgor = 'table[*data ^ (%s)crc] ^ (crc >> 8)'
        else:
            shift = 8 * (self.digest_size - 1)
            crcAlgor = 'table[*data ^ (%%s)(crc >> %d)] ^ (crc << 8)' % shift
        fmt = '0x%%0%dX' % (2 * self.digest_size)
        if self.digest_size <= 4:
            fmt = fmt + 'U,'
        else:
            fmt = fmt + 'ULL,'
        n = {1: 8, 2: 8, 3: 4, 4: 4, 8: 2}[self.digest_size]
        lst = []
        for i, val in enumerate(self.table):
            if i % n == 0:
                lst.append('\n    ')
            lst.append(fmt % val)
        poly = 'polynomial: 0x%X' % self.poly
        if self.reverse:
            poly = poly + ', bit reverse algorithm'
        if self.xorOut:
            preCondition = '\n    crc = crc ^ %s;' % (fmt[:-1] % self.xorOut)
            postCondition = preCondition
        else:
            preCondition = ''
            postCondition = ''
        if self.digest_size == 3:
            if self.reverse:
                preCondition += '\n    crc = crc & 0xFFFFFFU;'
            else:
                postCondition += '\n    crc = crc & 0xFFFFFFU;'
        parms = {'dataType': dataType, 'crcType': crcType, 'name': functionName, 'crcAlgor': crcAlgor % dataType, 'crcTable': ''.join(lst), 'poly': poly, 'preCondition': preCondition, 'postCondition': postCondition}
        out.write(_codeTemplate % parms)