import pytest
import traitlets
from bokeh.core.has_props import _default_resolver
from bokeh.model import Model
from panel.layout import Row
from panel.pane.ipywidget import Reacton
from panel.tests.util import serve_component, wait_until
@reacton.component
def ButtonClick():
    clicks, set_clicks = reacton.use_state(0)

    def test_effect():
        runs.append(button)

        def cleanup():
            cleanups.append(button)
        return cleanup
    reacton.use_effect(test_effect, [])

    def my_click_handler():
        click.append(clicks + 1)
        set_clicks(clicks + 1)
    button = reacton.ipywidgets.Button(description=f'Clicked {clicks} times', on_click=my_click_handler)
    return button