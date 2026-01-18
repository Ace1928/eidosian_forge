import pathlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen
import glob
import pytest
def _get_files_to_check():
    for post_fix in POST_FIXES:
        for file in glob.glob('**/*' + post_fix, recursive=True):
            yield pathlib.Path(file)