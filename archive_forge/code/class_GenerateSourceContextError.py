import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
class GenerateSourceContextError(Exception):
    """An error occurred while trying to create the source context."""
    pass