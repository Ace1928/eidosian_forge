import param
import pytest
from panel.io.location import Location
from panel.io.state import state
from panel.tests.util import serve_and_request, wait_until
from panel.util import edit_readonly
class SyncParameterized(param.Parameterized):
    integer = param.Integer(default=None)
    string = param.String(default=None)