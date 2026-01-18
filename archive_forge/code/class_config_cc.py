from distutils.core import Command
from numpy.distutils import log
class config_cc(Command):
    """ Distutils command to hold user specified options
    to C/C++ compilers.
    """
    description = 'specify C/C++ compiler information'
    user_options = [('compiler=', None, 'specify C/C++ compiler type')]

    def initialize_options(self):
        self.compiler = None

    def finalize_options(self):
        log.info('unifing config_cc, config, build_clib, build_ext, build commands --compiler options')
        build_clib = self.get_finalized_command('build_clib')
        build_ext = self.get_finalized_command('build_ext')
        config = self.get_finalized_command('config')
        build = self.get_finalized_command('build')
        cmd_list = [self, config, build_clib, build_ext, build]
        for a in ['compiler']:
            l = []
            for c in cmd_list:
                v = getattr(c, a)
                if v is not None:
                    if not isinstance(v, str):
                        v = v.compiler_type
                    if v not in l:
                        l.append(v)
            if not l:
                v1 = None
            else:
                v1 = l[0]
            if len(l) > 1:
                log.warn('  commands have different --%s options: %s, using first in list as default' % (a, l))
            if v1:
                for c in cmd_list:
                    if getattr(c, a) is None:
                        setattr(c, a, v1)
        return

    def run(self):
        return