import logging
import sys
from typing import Optional, TextIO
from absl import flags
from absl import logging as absl_logging
from googleapiclient import model
def ConfigurePythonLogger(apilog: Optional[str]=None):
    """Sets up Python logger.

  Applications can configure logging however they want, but this
  captures one pattern of logging which seems useful when dealing with
  a single command line option for determining logging.

  Args:
    apilog: To log to sys.stdout, specify '', '-', '1', 'true', or 'stdout'. To
      log to sys.stderr, specify 'stderr'. To log to a file, specify the file
      path. Specify None to disable logging.
  """
    if apilog is None:
        logging.debug('There is no apilog flag so non-critical logging is disabled.')
        logging.disable(logging.CRITICAL)
    else:
        if apilog in ('', '-', '1', 'true', 'stdout'):
            _SetLogFile(sys.stdout)
        elif apilog == 'stderr':
            _SetLogFile(sys.stderr)
        elif apilog:
            _SetLogFile(open(apilog, 'w'))
        else:
            logging.basicConfig(level=logging.INFO)
        if hasattr(flags.FLAGS, 'dump_request_response'):
            flags.FLAGS.dump_request_response = True
        else:
            model.dump_request_response = True