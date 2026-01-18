import sys
from distutils.file_util import write_file
class install(old_install):
    sub_commands = old_install.sub_commands + [('install_clib', lambda x: True)]

    def finalize_options(self):
        old_install.finalize_options(self)
        self.install_lib = self.install_libbase

    def setuptools_run(self):
        """ The setuptools version of the .run() method.

        We must pull in the entire code so we can override the level used in the
        _getframe() call since we wrap this call by one more level.
        """
        from distutils.command.install import install as distutils_install
        if self.old_and_unmanageable or self.single_version_externally_managed:
            return distutils_install.run(self)
        caller = sys._getframe(3)
        caller_module = caller.f_globals.get('__name__', '')
        caller_name = caller.f_code.co_name
        if caller_module != 'distutils.dist' or caller_name != 'run_commands':
            distutils_install.run(self)
        else:
            self.do_egg_install()

    def run(self):
        if not have_setuptools:
            r = old_install.run(self)
        else:
            r = self.setuptools_run()
        if self.record:
            with open(self.record) as f:
                lines = []
                need_rewrite = False
                for l in f:
                    l = l.rstrip()
                    if ' ' in l:
                        need_rewrite = True
                        l = '"%s"' % l
                    lines.append(l)
            if need_rewrite:
                self.execute(write_file, (self.record, lines), "re-writing list of installed files to '%s'" % self.record)
        return r