import logging
from pyasn1 import __version__
from pyasn1 import error
from pyasn1.compat.octets import octs2ints
class Debug(object):
    defaultPrinter = Printer()

    def __init__(self, *flags, **options):
        self._flags = flagNone
        if 'loggerName' in options:
            self._printer = Printer(logger=logging.getLogger(options['loggerName']), handler=NullHandler())
        elif 'printer' in options:
            self._printer = options.get('printer')
        else:
            self._printer = self.defaultPrinter
        self._printer('running pyasn1 %s, debug flags %s' % (__version__, ', '.join(flags)))
        for flag in flags:
            inverse = flag and flag[0] in ('!', '~')
            if inverse:
                flag = flag[1:]
            try:
                if inverse:
                    self._flags &= ~flagMap[flag]
                else:
                    self._flags |= flagMap[flag]
            except KeyError:
                raise error.PyAsn1Error('bad debug flag %s' % flag)
            self._printer("debug category '%s' %s" % (flag, inverse and 'disabled' or 'enabled'))

    def __str__(self):
        return 'logger %s, flags %x' % (self._printer, self._flags)

    def __call__(self, msg):
        self._printer(msg)

    def __and__(self, flag):
        return self._flags & flag

    def __rand__(self, flag):
        return flag & self._flags