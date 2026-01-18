import glob
import fnmatch
import string
import json
import os
import os.path as op
import shutil
import subprocess
import re
import copy
import tempfile
from os.path import join, dirname
from warnings import warn
from .. import config, logging
from ..utils.filemanip import (
from ..utils.misc import human_order_sorted, str2bool
from .base import (
def _get_files_over_ssh(self, template):
    """Get the files matching template over an SSH connection."""
    client = self._get_ssh_client()
    sftp = client.open_sftp()
    sftp.chdir(self.inputs.base_directory)
    template_dir = os.path.dirname(template)
    template_base = os.path.basename(template)
    every_file_in_dir = sftp.listdir(template_dir)
    if self.inputs.template_expression == 'fnmatch':
        outfiles = fnmatch.filter(every_file_in_dir, template_base)
    elif self.inputs.template_expression == 'regexp':
        regexp = re.compile(template_base)
        outfiles = list(filter(regexp.match, every_file_in_dir))
    else:
        raise ValueError('template_expression value invalid')
    if len(outfiles) == 0:
        msg = 'Output template: %s returned no files' % template
        if self.inputs.raise_on_empty:
            raise IOError(msg)
        else:
            warn(msg)
        outfiles = None
    else:
        if self.inputs.sort_filelist:
            outfiles = human_order_sorted(outfiles)
        if self.inputs.download_files:
            files_to_download = copy.copy(outfiles)
            for file_to_download in files_to_download:
                related_to_current = get_related_files(file_to_download, include_this_file=False)
                existing_related_not_downloading = [f for f in related_to_current if f in every_file_in_dir and f not in files_to_download]
                files_to_download.extend(existing_related_not_downloading)
            for f in files_to_download:
                try:
                    sftp.get(os.path.join(template_dir, f), f)
                except IOError:
                    iflogger.info('remote file %s not found' % f)
        outfiles = simplify_list(outfiles)
    return outfiles