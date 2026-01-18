from dash.testing.browser import Browser
class DashRComposite(Browser):

    def __init__(self, server, **kwargs):
        super().__init__(**kwargs)
        self.server = server

    def start_server(self, app, cwd=None):
        self.server(app, cwd=cwd)
        self.server_url = self.server.url