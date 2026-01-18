import param
import pytest
from playwright.sync_api import expect
from panel.reactive import ReactiveHTML
from panel.tests.util import serve_component, wait_until
class ReactiveLiteral(ReactiveHTML):
    value = param.String()
    _template = '\n    <div class="reactive">{{value}}</div>\n    '