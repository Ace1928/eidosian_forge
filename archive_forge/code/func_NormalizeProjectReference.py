from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import datetime
import hashlib
import json
import logging
import os
import random
import re
import string
import sys
import time
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
from absl import flags
import googleapiclient
import httplib2
from utils import bq_api_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
def NormalizeProjectReference(id_fallbacks: NamedTuple('IDS', [('project_id', Optional[str])]), reference: bq_id_utils.ApiClientHelper.ProjectReference) -> bq_id_utils.ApiClientHelper.ProjectReference:
    if reference is None:
        try:
            return GetProjectReference(id_fallbacks=id_fallbacks)
        except bq_error.BigqueryClientError as e:
            raise bq_error.BigqueryClientError('Project reference or a default project is required') from e
    return reference