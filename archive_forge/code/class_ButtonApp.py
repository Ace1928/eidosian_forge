import pytest
from playwright.sync_api import expect
from panel.pane import Textual
from panel.tests.util import serve_component, wait_until
class ButtonApp(App):

    def compose(self):
        yield Button('Default')

    def on_button_pressed(self, event: Button.Pressed) -> None:
        clicks.append(event)