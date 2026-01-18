import os
from configparser import ConfigParser
from distutils import log
from distutils.command.build_py import build_py
from distutils.command.install_scripts import install_scripts
from distutils.version import LooseVersion
from os.path import join as pjoin
from os.path import split as psplit
from os.path import splitext
class MyBuildPy(build_cmd):
    """Subclass to write commit data into installation tree"""

    def run(self):
        build_cmd.run(self)
        import subprocess
        proc = subprocess.Popen('git rev-parse --short HEAD', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        repo_commit, _ = proc.communicate()
        repo_commit = str(repo_commit)
        cfg_parser = ConfigParser()
        cfg_parser.read(pjoin(pkg_dir, 'COMMIT_INFO.txt'))
        cfg_parser.set('commit hash', 'install_hash', repo_commit)
        out_pth = pjoin(self.build_lib, pkg_dir, 'COMMIT_INFO.txt')
        cfg_parser.write(open(out_pth, 'wt'))