import shutil
import glob
import os
import sys
import tempfile
class appsi_build_ext(build_ext):

    def run(self):
        basedir = os.path.abspath(os.path.curdir)
        if self.inplace:
            tmpdir = os.path.join(this_file_dir(), 'cmodel')
        else:
            tmpdir = os.path.abspath(tempfile.mkdtemp())
        print("Building in '%s'" % tmpdir)
        os.chdir(tmpdir)
        try:
            super(appsi_build_ext, self).run()
            if not self.inplace:
                library = glob.glob('build/*/appsi_cmodel.*')[0]
                target = os.path.join(PYOMO_CONFIG_DIR, 'lib', 'python%s.%s' % sys.version_info[:2], 'site-packages', '.')
                if not os.path.exists(target):
                    os.makedirs(target)
                shutil.copy(library, target)
        finally:
            os.chdir(basedir)
            if not self.inplace:
                shutil.rmtree(tmpdir, onerror=handleReadonly)