import os
from copy import deepcopy
from nibabel import load
import numpy as np
from ... import logging
from ...utils import spm_docs as sd
from ..base import (
from ..base.traits_extension import NoDefaultSpecified
from ..matlab import MatlabCommand
from ...external.due import due, Doi, BibTeX
def _generate_job(self, prefix='', contents=None):
    """Recursive function to generate spm job specification as a string

        Parameters
        ----------
        prefix : string
            A string that needs to get
        contents : dict
            A non-tuple Python structure containing spm job
            information gets converted to an appropriate sequence of
            matlab commands.

        """
    jobstring = ''
    if contents is None:
        return jobstring
    if isinstance(contents, list):
        for i, value in enumerate(contents):
            if prefix.endswith(')'):
                newprefix = '%s,%d)' % (prefix[:-1], i + 1)
            else:
                newprefix = '%s(%d)' % (prefix, i + 1)
            jobstring += self._generate_job(newprefix, value)
        return jobstring
    if isinstance(contents, dict):
        for key, value in list(contents.items()):
            newprefix = '%s.%s' % (prefix, key)
            jobstring += self._generate_job(newprefix, value)
        return jobstring
    if isinstance(contents, np.ndarray):
        if contents.dtype == np.dtype(object):
            if prefix:
                jobstring += '%s = {...\n' % prefix
            else:
                jobstring += '{...\n'
            for i, val in enumerate(contents):
                if isinstance(val, np.ndarray):
                    jobstring += self._generate_job(prefix=None, contents=val)
                elif isinstance(val, list):
                    items_format = []
                    for el in val:
                        items_format += ['{}' if not isinstance(el, (str, bytes)) else "'{}'"]
                    val_format = ', '.join(items_format).format
                    jobstring += '[{}];...\n'.format(val_format(*val))
                elif isinstance(val, (str, bytes)):
                    jobstring += "'{}';...\n".format(val)
                else:
                    jobstring += '%s;...\n' % str(val)
            jobstring += '};\n'
        else:
            for i, val in enumerate(contents):
                for field in val.dtype.fields:
                    if prefix:
                        newprefix = '%s(%d).%s' % (prefix, i + 1, field)
                    else:
                        newprefix = '(%d).%s' % (i + 1, field)
                    jobstring += self._generate_job(newprefix, val[field])
        return jobstring
    if isinstance(contents, (str, bytes)):
        jobstring += "%s = '%s';\n" % (prefix, contents)
        return jobstring
    jobstring += '%s = %s;\n' % (prefix, str(contents))
    return jobstring