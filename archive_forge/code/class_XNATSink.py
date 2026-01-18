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
class XNATSink(LibraryBaseInterface, IOBase):
    """Generic datasink module that takes a directory containing a
    list of nifti files and provides a set of structured output
    fields.
    """
    input_spec = XNATSinkInputSpec
    _pkg = 'pyxnat'

    def _list_outputs(self):
        """Execute this module."""
        import pyxnat
        cache_dir = self.inputs.cache_dir or tempfile.gettempdir()
        if self.inputs.config:
            xnat = pyxnat.Interface(config=self.inputs.config)
        else:
            xnat = pyxnat.Interface(self.inputs.server, self.inputs.user, self.inputs.pwd, cache_dir)
        if self.inputs.share:
            subject_id = self.inputs.subject_id
            result = xnat.select('xnat:subjectData', ['xnat:subjectData/PROJECT', 'xnat:subjectData/SUBJECT_ID']).where('xnat:subjectData/SUBJECT_ID = %s AND' % subject_id)
            if result.data and isinstance(result.data[0], dict):
                result = result.data[0]
                shared = xnat.select('/project/%s/subject/%s' % (self.inputs.project_id, self.inputs.subject_id))
                if not shared.exists():
                    share_project = xnat.select('/project/%s' % self.inputs.project_id)
                    if not share_project.exists():
                        share_project.insert()
                    subject = xnat.select('/project/%(project)s/subject/%(subject_id)s' % result)
                    subject.share(str(self.inputs.project_id))
        uri_template_args = dict(project_id=quote_id(self.inputs.project_id), subject_id=self.inputs.subject_id, experiment_id=quote_id(self.inputs.experiment_id))
        if self.inputs.share:
            uri_template_args['original_project'] = result['project']
        if self.inputs.assessor_id:
            uri_template_args['assessor_id'] = quote_id(self.inputs.assessor_id)
        elif self.inputs.reconstruction_id:
            uri_template_args['reconstruction_id'] = quote_id(self.inputs.reconstruction_id)
        for key, files in list(self.inputs._outputs.items()):
            for name in ensure_list(files):
                if isinstance(name, list):
                    for i, file_name in enumerate(name):
                        push_file(self, xnat, file_name, '%s_' % i + key, uri_template_args)
                else:
                    push_file(self, xnat, name, key, uri_template_args)