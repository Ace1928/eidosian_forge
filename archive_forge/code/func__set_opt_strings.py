import sys, os
import textwrap
def _set_opt_strings(self, opts):
    for opt in opts:
        if len(opt) < 2:
            raise OptionError('invalid option string %r: must be at least two characters long' % opt, self)
        elif len(opt) == 2:
            if not (opt[0] == '-' and opt[1] != '-'):
                raise OptionError('invalid short option string %r: must be of the form -x, (x any non-dash char)' % opt, self)
            self._short_opts.append(opt)
        else:
            if not (opt[0:2] == '--' and opt[2] != '-'):
                raise OptionError('invalid long option string %r: must start with --, followed by non-dash' % opt, self)
            self._long_opts.append(opt)