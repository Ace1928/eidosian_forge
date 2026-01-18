import logging
import json
import re
from datetime import date, datetime, time, timezone
import traceback
import importlib
from typing import Any, Dict, Optional, Union, List, Tuple
from inspect import istraceback
from collections import OrderedDict
def _perform_rename_log_fields(self, log_record):
    for old_field_name, new_field_name in self.rename_fields.items():
        log_record[new_field_name] = log_record[old_field_name]
        del log_record[old_field_name]