from ctypes import Structure, c_int, c_byte
class action_t(Structure):
    _fields_ = [('action', c_int), ('length', c_int), ('value', c_byte * _VALUE_BUFFER_SIZE)]