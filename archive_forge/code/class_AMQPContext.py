import pprint
import click
from amqp import Connection, Message
from click_repl import register_repl
from celery.bin.base import handle_preload_options
class AMQPContext:

    def __init__(self, cli_context):
        self.cli_context = cli_context
        self.connection = self.cli_context.app.connection()
        self.channel = None
        self.reconnect()

    @property
    def app(self):
        return self.cli_context.app

    def respond(self, retval):
        if isinstance(retval, str):
            self.cli_context.echo(retval)
        else:
            self.cli_context.echo(pprint.pformat(retval))

    def echo_error(self, exception):
        self.cli_context.error(f'{self.cli_context.ERROR}: {exception}')

    def echo_ok(self):
        self.cli_context.echo(self.cli_context.OK)

    def reconnect(self):
        if self.connection:
            self.connection.close()
        else:
            self.connection = self.cli_context.app.connection()
        self.cli_context.echo(f'-> connecting to {self.connection.as_uri()}.')
        try:
            self.connection.connect()
        except (ConnectionRefusedError, ConnectionResetError) as e:
            self.echo_error(e)
        else:
            self.cli_context.secho('-> connected.', fg='green', bold=True)
            self.channel = self.connection.default_channel