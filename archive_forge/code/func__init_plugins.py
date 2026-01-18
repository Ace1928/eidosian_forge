from .exception import MultipleMatches
from .exception import NoMatches
from .named import NamedExtensionManager
def _init_plugins(self, extensions):
    super(DriverManager, self)._init_plugins(extensions)
    if not self.extensions:
        name = self._names[0]
        raise NoMatches('No %r driver found, looking for %r' % (self.namespace, name))
    if len(self.extensions) > 1:
        discovered_drivers = ','.join((e.entry_point_target for e in self.extensions))
        raise MultipleMatches('Multiple %r drivers found: %s' % (self.namespace, discovered_drivers))