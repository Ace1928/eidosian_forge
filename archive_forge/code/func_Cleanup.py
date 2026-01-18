import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
def Cleanup():
    for c in cleanups:
        c()