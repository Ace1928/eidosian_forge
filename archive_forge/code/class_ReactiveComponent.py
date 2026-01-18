import param
import pytest
from playwright.sync_api import expect
from panel.reactive import ReactiveHTML
from panel.tests.util import serve_component, wait_until
class ReactiveComponent(ReactiveHTML):
    count = param.Integer(default=0)
    event = param.Event()
    _template = '\n    <div id="reactive" class="reactive" onclick="${script(\'click\')}"></div>\n    '
    _scripts = {'render': 'data.count += 1; reactive.innerText = `${data.count}`;', 'click': 'data.count += 1; reactive.innerText = `${data.count}`;', 'event': 'data.count += 1; reactive.innerText = `${data.count}`;'}