import json
import os
import re
import sys
import numpy as np
def CamelCaseToSnakeCase(camel_case_input):
    """Converts an identifier in CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', '\\1_\\2', camel_case_input)
    return re.sub('([a-z0-9])([A-Z])', '\\1_\\2', s1).lower()