import os
import sys
def add_cffi_module(dist, mod_spec):
    from cffi.api import FFI
    if not isinstance(mod_spec, basestring):
        error("argument to 'cffi_modules=...' must be a str or a list of str, not %r" % (type(mod_spec).__name__,))
    mod_spec = str(mod_spec)
    try:
        build_file_name, ffi_var_name = mod_spec.split(':')
    except ValueError:
        error("%r must be of the form 'path/build.py:ffi_variable'" % (mod_spec,))
    if not os.path.exists(build_file_name):
        ext = ''
        rewritten = build_file_name.replace('.', '/') + '.py'
        if os.path.exists(rewritten):
            ext = ' (rewrite cffi_modules to [%r])' % (rewritten + ':' + ffi_var_name,)
        error('%r does not name an existing file%s' % (build_file_name, ext))
    mod_vars = {'__name__': '__cffi__', '__file__': build_file_name}
    execfile(build_file_name, mod_vars)
    try:
        ffi = mod_vars[ffi_var_name]
    except KeyError:
        error('%r: object %r not found in module' % (mod_spec, ffi_var_name))
    if not isinstance(ffi, FFI):
        ffi = ffi()
    if not isinstance(ffi, FFI):
        error('%r is not an FFI instance (got %r)' % (mod_spec, type(ffi).__name__))
    if not hasattr(ffi, '_assigned_source'):
        error('%r: the set_source() method was not called' % (mod_spec,))
    module_name, source, source_extension, kwds = ffi._assigned_source
    if ffi._windows_unicode:
        kwds = kwds.copy()
        ffi._apply_windows_unicode(kwds)
    if source is None:
        _add_py_module(dist, ffi, module_name)
    else:
        _add_c_module(dist, ffi, module_name, source, source_extension, kwds)