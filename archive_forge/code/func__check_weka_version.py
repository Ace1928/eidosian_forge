import os
import re
import subprocess
import tempfile
import time
import zipfile
from sys import stdin
from nltk.classify.api import ClassifierI
from nltk.internals import config_java, java
from nltk.probability import DictionaryProbDist
def _check_weka_version(jar):
    try:
        zf = zipfile.ZipFile(jar)
    except (SystemExit, KeyboardInterrupt):
        raise
    except:
        return None
    try:
        try:
            return zf.read('weka/core/version.txt')
        except KeyError:
            return None
    finally:
        zf.close()