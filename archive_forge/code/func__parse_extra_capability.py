from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def _parse_extra_capability(self, args):
    if self.NAME in args[self.EXTRA_CAPABILITY]:
        del args[self.EXTRA_CAPABILITY][self.NAME]
    args.update(args[self.EXTRA_CAPABILITY])
    args.pop(self.EXTRA_CAPABILITY)
    return args