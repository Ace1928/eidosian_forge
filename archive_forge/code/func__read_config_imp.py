import sys
import re
import os
from configparser import RawConfigParser
def _read_config_imp(filenames, dirs=None):

    def _read_config(f):
        meta, vars, sections, reqs = parse_config(f, dirs)
        for rname, rvalue in reqs.items():
            nmeta, nvars, nsections, nreqs = _read_config(pkg_to_filename(rvalue))
            for k, v in nvars.items():
                if not k in vars:
                    vars[k] = v
            for oname, ovalue in nsections[rname].items():
                if ovalue:
                    sections[rname][oname] += ' %s' % ovalue
        return (meta, vars, sections, reqs)
    meta, vars, sections, reqs = _read_config(filenames)
    if not 'pkgdir' in vars and 'pkgname' in vars:
        pkgname = vars['pkgname']
        if not pkgname in sys.modules:
            raise ValueError('You should import %s to get information on %s' % (pkgname, meta['name']))
        mod = sys.modules[pkgname]
        vars['pkgdir'] = _escape_backslash(os.path.dirname(mod.__file__))
    return LibraryInfo(name=meta['name'], description=meta['description'], version=meta['version'], sections=sections, vars=VariableSet(vars))