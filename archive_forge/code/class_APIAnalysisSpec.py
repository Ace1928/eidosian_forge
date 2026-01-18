import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
class APIAnalysisSpec:
    """This class defines how `AnalysisResult`s should be generated.

  It specifies how to map imports and symbols to `AnalysisResult`s.

  This class must provide the following fields:

  * `symbols_to_detect`: maps function names to `AnalysisResult`s
  * `imports_to_detect`: maps imports represented as (full module name, alias)
    tuples to `AnalysisResult`s
    notifications)

  For an example, see `TFAPIImportAnalysisSpec`.
  """