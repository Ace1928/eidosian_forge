from dash.testing.browser import Browser
class DashComposite(Browser):

    def __init__(self, server, **kwargs):
        super().__init__(**kwargs)
        self.server = server

    def start_server(self, app, navigate=True, **kwargs):
        """Start the local server with app."""
        self.server(app, **kwargs)
        if navigate:
            self.server_url = self.server.url