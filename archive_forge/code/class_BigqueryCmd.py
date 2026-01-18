import logging
import os
import pdb
import shlex
import sys
import traceback
import types
from absl import app
from absl import flags
import googleapiclient
import bq_flags
import bq_utils
from utils import bq_error
from utils import bq_logging
from utils import bq_processor_utils
from pyglib import appcommands
class BigqueryCmd(NewCmd):
    """Bigquery-specific NewCmd wrapper."""

    def _NeedsInit(self) -> bool:
        """Returns true if this command requires the init command before running.

    Subclasses will override for any exceptional cases.
    """
        return not _UseServiceAccount() and (not (os.path.exists(bq_utils.GetBigqueryRcFilename()) or os.path.exists(FLAGS.credential_file)))

    def Run(self, argv):
        """Bigquery commands run `init` before themselves if needed."""
        if FLAGS.debug_mode:
            cmd_flags = [FLAGS[f].serialize().strip() for f in FLAGS if FLAGS[f].present]
            print(' '.join(sorted(set((f for f in cmd_flags if f)))))
        bq_logging.ConfigureLogging(bq_flags.APILOG.value)
        logging.debug('In BigqueryCmd.Run: %s', argv)
        if self._NeedsInit():
            appcommands.GetCommandByName('init').Run(['init'])
        return super(BigqueryCmd, self).Run(argv)

    def RunSafely(self, args, kwds):
        """Run this command, printing information about any exceptions raised."""
        logging.debug('In BigqueryCmd.RunSafely: %s, %s', args, kwds)
        try:
            return_value = self.RunWithArgs(*args, **kwds)
        except BaseException as e:
            return bq_utils.ProcessError(e, name=self._command_name)
        return return_value

    def PrintJobStartInfo(self, job):
        """Print a simple status line."""
        if FLAGS.format in ['prettyjson', 'json']:
            bq_utils.PrintFormattedJsonObject(job)
        else:
            reference = bq_processor_utils.ConstructObjectReference(job)
            print('Successfully started %s %s' % (self._command_name, reference))

    def _ProcessCommandRc(self, fv):
        bq_utils.ProcessBigqueryrcSection(self._command_name, fv)