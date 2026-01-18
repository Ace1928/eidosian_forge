import glob
import logging
import os
import shutil
import subprocess
import time
from apache_beam.io import gcsio
Writes data to a file.

  Supports local and Google Cloud Storage.

  Args:
    path: output file path.
    data: data to write to file.
  