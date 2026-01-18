import sys, os
import textwrap
def _process_long_opt(self, rargs, values):
    arg = rargs.pop(0)
    if '=' in arg:
        opt, next_arg = arg.split('=', 1)
        rargs.insert(0, next_arg)
        had_explicit_value = True
    else:
        opt = arg
        had_explicit_value = False
    opt = self._match_long_opt(opt)
    option = self._long_opt[opt]
    if option.takes_value():
        nargs = option.nargs
        if len(rargs) < nargs:
            self.error(ngettext('%(option)s option requires %(number)d argument', '%(option)s option requires %(number)d arguments', nargs) % {'option': opt, 'number': nargs})
        elif nargs == 1:
            value = rargs.pop(0)
        else:
            value = tuple(rargs[0:nargs])
            del rargs[0:nargs]
    elif had_explicit_value:
        self.error(_('%s option does not take a value') % opt)
    else:
        value = None
    option.process(opt, value, values, self)