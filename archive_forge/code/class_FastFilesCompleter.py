import argparse
from glob import glob
import os
import sys
from typing import Any
from typing import List
from typing import Optional
class FastFilesCompleter:
    """Fast file completer class."""

    def __init__(self, directories: bool=True) -> None:
        self.directories = directories

    def __call__(self, prefix: str, **kwargs: Any) -> List[str]:
        if os.sep in prefix[1:]:
            prefix_dir = len(os.path.dirname(prefix) + os.sep)
        else:
            prefix_dir = 0
        completion = []
        globbed = []
        if '*' not in prefix and '?' not in prefix:
            if not prefix or prefix[-1] == os.sep:
                globbed.extend(glob(prefix + '.*'))
            prefix += '*'
        globbed.extend(glob(prefix))
        for x in sorted(globbed):
            if os.path.isdir(x):
                x += '/'
            completion.append(x[prefix_dir:])
        return completion