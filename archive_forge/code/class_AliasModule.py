import importlib
import sys
class AliasModule(object):
    """Wrapper around Python module to allow lazy loading."""

    def __init__(self, name, members=None):
        """Initialize a module without loading it.

        Parameters
        ----------
        name : str
            Module name
        members : list[str, dict]
            List of submodules to be loaded as AliasModules. If a dict, the submodule
            is loaded with subsubmodules corresponding to the dictionary values;
            if a string, the submodule has no subsubmodules.
        """
        super_setattr = super().__setattr__
        if members is None:
            members = []
        builtin_members = ['__class__', '__doc__']
        super_setattr('__module_name__', name)
        submodules = []
        for member in members:
            if isinstance(member, dict):
                for submodule, submembers in member.items():
                    super_setattr(submodule, AliasModule('{}.{}'.format(name, submodule), submembers))
                    submodules.append(submodule)
            else:
                super_setattr(member, AliasModule('{}.{}'.format(name, member)))
                submodules.append(member)
        super_setattr('__submodules__', submodules)
        super_setattr('__builtin_members__', builtin_members)

    @property
    def __loaded_module__(self):
        """Load the module, or retrieve it if already loaded."""
        super_getattr = super().__getattribute__
        name = super_getattr('__module_name__')
        try:
            return sys.modules[name]
        except KeyError:
            importlib.import_module(name)
            return sys.modules[name]

    def __getattribute__(self, attr):
        """Access AliasModule members."""
        super_getattr = super().__getattribute__
        if attr in super_getattr('__submodules__'):
            return super_getattr(attr)
        elif attr in super_getattr('__builtin_members__'):
            if super_getattr('__module_name__') in sys.modules:
                return getattr(super_getattr('__loaded_module__'), attr)
            else:
                return super_getattr(attr)
        else:
            return getattr(super_getattr('__loaded_module__'), attr)

    def __setattr__(self, name, value):
        """Allow monkey-patching.

        Gives easy access to AliasModule members to avoid recursionerror.
        """
        super_getattr = super().__getattribute__
        return setattr(super_getattr('__loaded_module__'), name, value)