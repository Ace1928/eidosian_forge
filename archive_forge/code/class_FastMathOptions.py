from abc import ABCMeta, abstractmethod
class FastMathOptions(AbstractOptionValue):
    """
    Options for controlling fast math optimization.
    """

    def __init__(self, value):
        valid_flags = {'fast', 'nnan', 'ninf', 'nsz', 'arcp', 'contract', 'afn', 'reassoc'}
        if isinstance(value, FastMathOptions):
            self.flags = value.flags.copy()
        elif value is True:
            self.flags = {'fast'}
        elif value is False:
            self.flags = set()
        elif isinstance(value, set):
            invalid = value - valid_flags
            if invalid:
                raise ValueError('Unrecognized fastmath flags: %s' % invalid)
            self.flags = value
        elif isinstance(value, dict):
            invalid = set(value.keys()) - valid_flags
            if invalid:
                raise ValueError('Unrecognized fastmath flags: %s' % invalid)
            self.flags = {v for v, enable in value.items() if enable}
        else:
            msg = 'Expected fastmath option(s) to be either a bool, dict or set'
            raise ValueError(msg)

    def __bool__(self):
        return bool(self.flags)
    __nonzero__ = __bool__

    def encode(self) -> str:
        return str(self.flags)

    def __eq__(self, other):
        if type(other) is type(self):
            return self.flags == other.flags
        return NotImplemented