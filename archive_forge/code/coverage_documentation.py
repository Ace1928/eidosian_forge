import os
import coverage
from kivy.lang.parser import Parser
Kivy coverage plugin
=======================

This provides a coverage plugin to measure code execution in kv files. To use,
create and add::

    [run]
    plugins =
        kivy.tools.coverage

to the ``.coveragerc`` file in the root of your project. Or::

    [coverage:run]
    plugins =
        kivy.tools.coverage

in ``setup.cfg``.

Then you can test your project with e.g. ``pip install coverage`` followed by
``coverage run --source=./ kivy_app.py`` and ``coverage report``.

Or to use with pytest, ``pip install pytest-cov`` followed by something like
``pytest --cov=./ .``

TODO: Expand kv statements measured.

Currently, it ignores rules such as Widget creation or graphics object
creation from being measured. Similarly for import statements.

KV code created as strings within a python file is also not measured. To
support the above, deeper changes will be required.
