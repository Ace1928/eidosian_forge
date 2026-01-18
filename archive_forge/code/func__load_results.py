from collections import OrderedDict, defaultdict
import os
import os.path as op
from pathlib import Path
import shutil
import socket
from copy import deepcopy
from glob import glob
from logging import INFO
from tempfile import mkdtemp
from ... import config, logging
from ...utils.misc import flatten, unflatten, str2bool, dict_diff
from ...utils.filemanip import (
from ...interfaces.base import (
from ...interfaces.base.specs import get_filecopy_info
from .utils import (
from .base import EngineBase
def _load_results(self):
    cwd = self.output_dir()
    try:
        result = _load_resultfile(op.join(cwd, 'result_%s.pklz' % self.name))
    except (traits.TraitError, EOFError):
        logger.debug('Error populating inputs/outputs, (re)aggregating results...')
    except (AttributeError, ImportError) as err:
        logger.debug('attribute error: %s probably using different trait pickled file', str(err))
        old_inputs = loadpkl(op.join(cwd, '_inputs.pklz'))
        self.inputs.trait_set(**old_inputs)
    else:
        return result
    if not isinstance(self, MapNode):
        self._copyfiles_to_wd(linksonly=True)
        aggouts = self._interface.aggregate_outputs(needed_outputs=self.needed_outputs)
        runtime = Bunch(cwd=cwd, returncode=0, environ=dict(os.environ), hostname=socket.gethostname())
        result = InterfaceResult(interface=self._interface.__class__, runtime=runtime, inputs=self._interface.inputs.get_traitsfree(), outputs=aggouts)
        _save_resultfile(result, cwd, self.name, rebase=str2bool(self.config['execution']['use_relative_paths']))
    else:
        logger.debug('aggregating mapnode results')
        result = self._run_interface()
    return result