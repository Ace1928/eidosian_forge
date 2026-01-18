import numpy as np
from scipy._lib import doccer
from . import _byteordercodes as boc
class MatFileReader:
    """ Base object for reading mat files

    To make this class functional, you will need to override the
    following methods:

    matrix_getter_factory   - gives object to fetch next matrix from stream
    guess_byte_order        - guesses file byte order from file
    """

    @docfiller
    def __init__(self, mat_stream, byte_order=None, mat_dtype=False, squeeze_me=False, chars_as_strings=True, matlab_compatible=False, struct_as_record=True, verify_compressed_data_integrity=True, simplify_cells=False):
        """
        Initializer for mat file reader

        mat_stream : file-like
            object with file API, open for reading
    %(load_args)s
        """
        self.mat_stream = mat_stream
        self.dtypes = {}
        if not byte_order:
            byte_order = self.guess_byte_order()
        else:
            byte_order = boc.to_numpy_code(byte_order)
        self.byte_order = byte_order
        self.struct_as_record = struct_as_record
        if matlab_compatible:
            self.set_matlab_compatible()
        else:
            self.squeeze_me = squeeze_me
            self.chars_as_strings = chars_as_strings
            self.mat_dtype = mat_dtype
        self.verify_compressed_data_integrity = verify_compressed_data_integrity
        self.simplify_cells = simplify_cells
        if simplify_cells:
            self.squeeze_me = True
            self.struct_as_record = False

    def set_matlab_compatible(self):
        """ Sets options to return arrays as MATLAB loads them """
        self.mat_dtype = True
        self.squeeze_me = False
        self.chars_as_strings = False

    def guess_byte_order(self):
        """ As we do not know what file type we have, assume native """
        return boc.native_code

    def end_of_stream(self):
        b = self.mat_stream.read(1)
        curpos = self.mat_stream.tell()
        self.mat_stream.seek(curpos - 1)
        return len(b) == 0