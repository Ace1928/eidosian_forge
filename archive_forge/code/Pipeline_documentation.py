from __future__ import absolute_import
import itertools
from time import time
from . import Errors
from . import DebugFlags
from . import Options
from .Errors import CompileError, InternalError, AbortError
from . import Naming

    Insert a new transform into the pipeline after or before an instance of
    the given class. e.g.

        pipeline = insert_into_pipeline(pipeline, transform,
                                        after=AnalyseDeclarationsTransform)
    