import logging
import os
import sys
import taskflow.engines
from taskflow.patterns import graph_flow as gf
from taskflow import task
import example_utils as eu  # noqa
class LinkTask(task.Task):
    """Pretends to link executable form several object files."""
    default_provides = 'executable'

    def __init__(self, executable_path, *args, **kwargs):
        super(LinkTask, self).__init__(*args, **kwargs)
        self._executable_path = executable_path

    def execute(self, **kwargs):
        object_filenames = list(kwargs.values())
        print('Linking executable %s from files %s' % (self._executable_path, ', '.join(object_filenames)))
        return self._executable_path