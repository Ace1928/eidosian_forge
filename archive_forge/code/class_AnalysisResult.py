import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
class AnalysisResult:
    """This class represents an analysis result and how it should be logged.

  This class must provide the following fields:

  * `log_level`: The log level to which this detection should be logged
  * `log_message`: The message that should be logged for this detection

  For an example, see `VersionedTFImport`.
  """