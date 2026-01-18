import sys, os
def _fake_import(fn, name):
    from importlib.util import spec_from_loader, module_from_spec
    from importlib.machinery import SourceFileLoader
    spec = spec_from_loader(name, SourceFileLoader(name, fn))
    module = module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError:
        raise ImportError('file %s not found' % ascii(fn))
    sys.modules[name] = module