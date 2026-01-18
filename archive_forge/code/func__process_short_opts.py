import sys, os
import textwrap
def _process_short_opts(self, rargs, values):
    arg = rargs.pop(0)
    stop = False
    i = 1
    for ch in arg[1:]:
        opt = '-' + ch
        option = self._short_opt.get(opt)
        i += 1
        if not option:
            raise BadOptionError(opt)
        if option.takes_value():
            if i < len(arg):
                rargs.insert(0, arg[i:])
                stop = True
            nargs = option.nargs
            if len(rargs) < nargs:
                self.error(ngettext('%(option)s option requires %(number)d argument', '%(option)s option requires %(number)d arguments', nargs) % {'option': opt, 'number': nargs})
            elif nargs == 1:
                value = rargs.pop(0)
            else:
                value = tuple(rargs[0:nargs])
                del rargs[0:nargs]
        else:
            value = None
        option.process(opt, value, values, self)
        if stop:
            break