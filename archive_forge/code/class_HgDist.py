from __future__ import annotations
import abc
import argparse
import gzip
import os
import sys
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import hashlib
import typing as T
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from mesonbuild.environment import Environment, detect_ninja
from mesonbuild.mesonlib import (MesonException, RealPathAction, get_meson_command, quiet_git,
from mesonbuild.msetup import add_arguments as msetup_argparse
from mesonbuild.wrap import wrap
from mesonbuild import mlog, build, coredata
from .scripts.meson_exe import run_exe
class HgDist(Dist):

    def have_dirty_index(self) -> bool:
        """Check whether there are uncommitted changes in hg"""
        out = subprocess.check_output(['hg', '-R', self.src_root, 'summary'])
        return b'commit: (clean)' not in out

    def create_dist(self, archives: T.List[str]) -> T.List[str]:
        if self.have_dirty_index():
            handle_dirty_opt(msg_uncommitted_changes, self.options.allow_dirty)
        if self.dist_scripts:
            mlog.warning('dist scripts are not supported in Mercurial projects')
        os.makedirs(self.dist_sub, exist_ok=True)
        tarname = os.path.join(self.dist_sub, self.dist_name + '.tar')
        xzname = tarname + '.xz'
        gzname = tarname + '.gz'
        zipname = os.path.join(self.dist_sub, self.dist_name + '.zip')
        subprocess.check_call(['hg', 'archive', '-R', self.src_root, '-S', '-t', 'tar', '-X', self.src_root + '/.hg[a-z]*', tarname])
        output_names = []
        if 'xztar' in archives:
            import lzma
            with lzma.open(xzname, 'wb') as xf, open(tarname, 'rb') as tf:
                shutil.copyfileobj(tf, xf)
            output_names.append(xzname)
        if 'gztar' in archives:
            with gzip.open(gzname, 'wb') as zf, open(tarname, 'rb') as tf:
                shutil.copyfileobj(tf, zf)
            output_names.append(gzname)
        os.unlink(tarname)
        if 'zip' in archives:
            subprocess.check_call(['hg', 'archive', '-R', self.src_root, '-S', '-t', 'zip', zipname])
            output_names.append(zipname)
        return output_names