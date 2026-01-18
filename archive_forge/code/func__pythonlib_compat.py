import sys
def _pythonlib_compat():
    """
    On Python 3.7 and earlier, distutils would include the Python
    library. See pypa/distutils#9.
    """
    from distutils import sysconfig
    if not sysconfig.get_config_var('Py_ENABLED_SHARED'):
        return
    yield 'python{}.{}{}'.format(sys.hexversion >> 24, sys.hexversion >> 16 & 255, sysconfig.get_config_var('ABIFLAGS'))