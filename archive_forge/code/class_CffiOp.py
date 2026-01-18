from .error import VerificationError
class CffiOp(object):

    def __init__(self, op, arg):
        self.op = op
        self.arg = arg

    def as_c_expr(self):
        if self.op is None:
            assert isinstance(self.arg, str)
            return '(_cffi_opcode_t)(%s)' % (self.arg,)
        classname = CLASS_NAME[self.op]
        return '_CFFI_OP(_CFFI_OP_%s, %s)' % (classname, self.arg)

    def as_python_bytes(self):
        if self.op is None and self.arg.isdigit():
            value = int(self.arg)
            if value >= 2 ** 31:
                raise OverflowError('cannot emit %r: limited to 2**31-1' % (self.arg,))
            return format_four_bytes(value)
        if isinstance(self.arg, str):
            raise VerificationError('cannot emit to Python: %r' % (self.arg,))
        return format_four_bytes(self.arg << 8 | self.op)

    def __str__(self):
        classname = CLASS_NAME.get(self.op, self.op)
        return '(%s %s)' % (classname, self.arg)