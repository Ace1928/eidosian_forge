import os
import shutil
from tempfile import mkdtemp
import pytest
import numpy
import py.path as pp
@pytest.fixture(autouse=True)
def _docdir(request):
    """Grabbed from https://stackoverflow.com/a/46991331"""
    doctest_plugin = request.config.pluginmanager.getplugin('doctest')
    if isinstance(request.node, doctest_plugin.DoctestItem):
        tmpdir = pp.local(data_dir)
        with tmpdir.as_cwd():
            yield
    else:
        yield